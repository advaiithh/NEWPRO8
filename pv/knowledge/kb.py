"""Offline general-knowledge base: lead paragraphs of Wikipedia, retrieved with static embeddings.

Two builds exist: `models/kb` (Simple English, 2023, small) and `models/kb_en` (English Wikipedia, October 2026 export, built by
scripts/build_kb_full.py). A profile chooses with `kb: {dir: ...}`.

Definition questions ("what is photosynthesis") and office-holder questions ("who is the chief minister of Tamil Nadu") are
answered by reading the article aloud; other factual questions pass the matching lead to the LLM as context.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from dataclasses import dataclass

import numpy as np
from model2vec import StaticModel
from rapidfuzz import fuzz, process

from pv.config import config, model

DEFINITION = [
    re.compile(r"^(?:what|who)\s+(?:is|are|was|were)\s+(?:an?\s+|the\s+)?(?P<t>[^?]+?)\s*\??$"),
    re.compile(r"^(?:what|who)'s\s+(?:an?\s+|the\s+)?(?P<t>[^?]+?)\s*\??$"),
    re.compile(r"^(?:tell me about|tell me more about|explain|define|describe|what do you know about|give me information (?:on|about))\s+(?:an?\s+|the\s+)?(?P<t>[^?]+?)\s*\??$"),
    re.compile(r"^what does\s+(?P<t>[^?]+?)\s+mean\s*\??$"),
]
# "who is the current chief minister of tamil nadu": an office-holder question (present tense)
OFFICE_Q = re.compile(r"^(?:who(?:'s| is| are)|which person is|name)\s+(?:the\s+)?(?:current\s+|present\s+|incumbent\s+|new\s+|latest\s+)?(?P<t>.+?)"
                      r"(?:\s+(?:right now|now|currently|today|at present|at the moment|these days))?\s*\??$")
OFFICE_WORDS = re.compile(r"\b(minister|president|governor|speaker|mayor|chief|prime|secretary|chairman|chairperson|leader|king|queen|emperor|"
                          r"pope|chancellor|director|commissioner|justice|ambassador|general|captain|champion|head|ceo|founder|owner)\b")
# "what is the capital of japan" is a fact about a topic, not a definition of the topic
NOT_DEFINITION = re.compile(r"\b(of|in|on|for|between|from|by|to|with|at)\b|\b(time|date|day|weather|price|score|temperature|meaning)\b")
ABBREV = {"cm": "chief minister", "pm": "prime minister", "dcm": "deputy chief minister", "tn": "tamil nadu", "uk": "united kingdom",
          "usa": "united states", "us": "united states", "uae": "united arab emirates", "un": "united nations",
          "ipl": "indian premier league", "epl": "premier league", "wc": "world cup", "f1": "formula one", "nba": "nba", "oscars": "academy awards"}
PRESENT = re.compile(r"\b(is|serves|serving|incumbent|currently)\b|\bsince\b", re.I)
NAME_START = re.compile(r"^(?!The |A |An |In |On |Of |General |Elections? )[A-Z][^\W\d_]+(?:[ .'\u2019-]+[A-Z][^\W\d_]*)+")
PAST = re.compile(r"\b(was|formerly|late|died|resigned|retired)\b|\buntil\b|\bfrom\b[^.]*\b(to|until)\b[^.]*\b(19|20)\d{2}\b|\bbetween\b[^.]*\band\b[^.]*\b(19|20)\d{2}\b|(?<!has )(?<!have )\bserved as\b(?![^.]*\bsince\b)", re.I)
WON_Q = re.compile(r"^who\s+(?:won|wins|became|has won|is winning|will win)\s+(?:the\s+)?(?P<t>.+?)\s*\??$")
PERSON = re.compile(r"\b(politician|statesman|stateswoman|businessman|businesswoman|diplomat|lawyer|judge|jurist|economist|bureaucrat|"
                    r"activist|journalist|academic|professor|scientist|banker|barrister|advocate|actor|actress|soldier|civil servant|"
                    r"monarch|cleric|priest|engineer|physician|surgeon|author|entrepreneur|administrator|trade unionist|military)\b", re.I)
# words a matched event title may add: a bare election/season descriptor, never "bid", "controversies", "qualification", "U-20", a country...
ALLOWED_EXTRA = {"legislative", "assembly", "general", "state", "presidential", "parliamentary", "lok", "sabha", "senate", "gubernatorial",
                 "mens", "final", "finals", "season", "edition", "the", "of", "and", "united", "states", "indian", "national", "federal"}
OFFICE_LEAD = {"the", "as", "a", "an", "been", "become", "becoming", "being", "current", "incumbent", "new", "interim", "caretaker", "first", "second", "third", "serving", "and", "also"}
HOLDER_HINT = re.compile(r"incumbent|currently|current|since|serves?|served|has been|is the|was sworn|elected", re.I)


@dataclass
class Hit:
    title: str
    lead: str
    score: float
    title_match: bool

    def snippet(self, max_words: int = 45, sentences: int = 2, prefer_holder: bool = False) -> str:
        parts = re.split(r"(?<=[.!?])\s+", self.lead.strip())
        if prefer_holder:                                   # lead the answer with the sentence that names the current holder
            holder = [p for p in parts if HOLDER_HINT.search(p) and re.search(r"[A-Z][a-z]+ [A-Z]", p)]
            parts = holder[:1] + [p for p in parts if p not in holder[:1]]
        out: list[str] = []
        for s_ in parts[:sentences]:
            if len(" ".join(out + [s_]).split()) > max_words and out:
                break
            out.append(s_)
        text = " ".join(out)
        words = text.split()
        return text if len(words) <= max_words else " ".join(words[:max_words]).rstrip(",;:") + "."


def clean_query(q: str) -> str:
    q = re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", q.lower())).strip()
    return " ".join(ABBREV.get(w, w) for w in q.split())


def topic_of(question: str) -> tuple[str, bool]:
    """(topic, is_definition_question)."""
    q = clean_query(question.rstrip(".! "))
    for pat in DEFINITION:
        if m := pat.match(q):
            t = m["t"].strip()
            return t, not NOT_DEFINITION.search(t) and 0 < len(t.split()) <= 5
    return q, False


def office_topic(question: str) -> str | None:
    """'who is the current CM of Tamil Nadu' -> 'chief minister of tamil nadu'; None when it is not an office question."""
    m = OFFICE_Q.match(clean_query(question.rstrip(".! ")))
    if not m:
        return None
    t = re.sub(r"^(the|a|an)\s+", "", m["t"].strip())
    return t if OFFICE_WORDS.search(t) and 1 < len(t.split()) <= 8 else None


class Knowledge:
    def __init__(self, rag_threshold: float = 0.5) -> None:
        d = model(config.get("kb", {}).get("dir", "kb"))
        if not (d / "emb.npy").exists():
            d = model("kb")
        self.dir = d
        self.emb = np.load(d / "emb.npy", mmap_mode="r")
        self.titles: list[str] = json.loads((d / "titles.json").read_text(encoding="utf-8"))
        self.db = sqlite3.connect(d / "leads.sqlite", check_same_thread=False) if (d / "leads.sqlite").exists() else None
        self.leads: list[str] = [] if self.db else json.loads((d / "leads.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        self.snapshot = meta.get("snapshot", "20231101")
        self.asof = dt.datetime.strptime(self.snapshot, "%Y%m%d").strftime("%B %Y")
        self.by_title = {clean_query(t): i for i, t in enumerate(self.titles)}
        self.model = StaticModel.from_pretrained(str(model("embed", "potion-base-8M")))
        self.rag_threshold = rag_threshold

    def lead(self, i: int) -> str:
        if self.db:
            return self.db.execute("SELECT lead FROM leads WHERE i = ?", (i,)).fetchone()[0]
        return self.leads[i]

    def _hit(self, i: int, score: float, exact: bool) -> Hit:
        return Hit(self.titles[i], self.lead(i), score, exact)

    def _embed(self, text: str) -> np.ndarray:
        v = self.model.encode([text])[0]
        return (v / max(float(np.linalg.norm(v)), 1e-9)).astype(np.float32)

    def _nearest(self, text: str) -> tuple[int, float]:
        q = self._embed(text)
        best, best_i = -1.0, 0
        for a in range(0, len(self.emb), 40000):
            s = self.emb[a:a + 40000].astype(np.float32) @ q
            j = int(np.argmax(s))
            if s[j] > best:
                best, best_i = float(s[j]), a + j
        return best_i, best

    def _nearest_k(self, text: str, k: int = 25) -> list[tuple[int, float]]:
        q = self._embed(text)
        scores = np.concatenate([self.emb[a:a + 40000].astype(np.float32) @ q for a in range(0, len(self.emb), 40000)])
        top = np.argpartition(-scores, k)[:k]
        return [(int(i), float(scores[i])) for i in top[np.argsort(-scores[top])]]

    def holder_sentence(self, office: str) -> tuple[Hit, str] | None:
        """The person whose own article says, in the present tense, that they hold `office`.

        Office articles describe the office, not the holder (the name sits in a side box that text dumps drop), so scan the leads
        for biographies such as "X is an Indian politician currently serving as the ninth chief minister of Tamil Nadu since 2026".
        Tense is judged only inside the clause that mentions the office ("former actor who is currently serving" is fine).
        """
        if not self.db:
            return None
        role, _, place = office.partition(" of ")
        place_words = [w for w in re.findall(r"[a-z]+", place) if w != "the"]
        # "<role> of [the/filler words] <place>": rejects "Principal Secretary to the Prime Minister", "Prime Minister's Questions", "president of <party>"
        filler = r"(?:(?:the|state|republic|union|national|capital|territory|federal|islamic|peoples|people's|democratic|kingdom|commonwealth|of)\s+){0,6}"
        head = r"\b" + r"[\s-]+".join(map(re.escape, role.split()))
        role_re = re.compile(head + r"\s+of\s+" + filler + r"\s+".join(map(re.escape, place_words)) + r"\b" if place_words else head + r"\b", re.I)
        sql = "SELECT i, lead FROM leads WHERE lower(replace(lead, '-', ' ')) LIKE ?" + " AND lower(lead) LIKE ?" * len(place_words)
        args = [f"%{role}%"] + [f"%{w}%" for w in place_words]
        best = None
        for i, lead in self.db.execute(sql + " LIMIT 4000", args):
            title = clean_query(self.titles[i])
            if title.startswith(("list of", "first ", "second ", "third ")) or "election" in title or "ministry" in title or fuzz.partial_ratio(title, office) >= 85:
                continue                                    # the office / government / election article itself, not a person
            for sent in re.split(r"(?<=[.!?])\s+", lead):
                if sent[:3].lower() in ("is ", "has", "was"):
                    sent = f"{self.titles[i]} {sent}"
                flat = sent.replace("-", " ")
                m = role_re.search(flat)
                if not m or not NAME_START.match(sent):
                    continue
                if not PERSON.search(flat):
                    continue                                # "Air Force One is the official call sign ..." is not a person
                if re.search(r"\b(to|of|for|at)\s+(the\s+)?$", flat[:m.start()], re.I):
                    continue                                # "Principal Secretary to the Prime Minister of India"
                before = flat[:m.start()].split()[-1:] or [""]
                if before[0].lower() not in OFFICE_LEAD and not re.match(r"^(\d+(st|nd|rd|th)|\w+th|\w+(first|second|third))$", before[0], re.I):
                    continue                                # "X, Chief Minister of ..." is fine; "Secretary to the Prime Minister" is not
                cut = max(flat.rfind(" who", 0, m.start()), flat.rfind(" that ", 0, m.start()), flat.rfind("; ", 0, m.start()))
                clause = flat[max(cut, 0): m.end() + 60]
                if PRESENT.search(clause) and not PAST.search(clause):
                    rank = 2 if re.search(r"\bcurrently\b", clause, re.I) else 1
                    if best is None or rank > best[0]:
                        best = (rank, i, sent)
                    break
        return (self._hit(best[1], 1.0, False), best[2]) if best else None

    def event_answer(self, question: str) -> tuple[Hit, str] | None:
        """'who won the 2026 Tamil Nadu election' -> the event article's lead, which states the result."""
        m = WON_Q.match(clean_query(question.rstrip(".! ")))
        if not m:
            return None
        topic = m["t"].strip()
        if not re.search(r"\b(election|elections|cup|trophy|championship|series|league|prize|award|awards|race|referendum|final|open|olympics|olympic|"
                         r"nobel|grammy|oscar|wimbledon|slam|grand prix|tournament|medal|title|bowl|masters|ballon|games|marathon|derby)\b", topic):
            return None
        year = re.search(r"\b(?:19|20)\d{2}\b", topic)
        if not year:
            return None                                 # "who won the World Cup" is ambiguous: only answer for a named edition
        want = set(topic.split())
        best: tuple[int, str] | None = None
        for key in self.by_title:                       # titles of that year whose words include every word asked about
            if year.group() not in key:
                continue
            words = set(key.split())
            extra = words - want
            if want <= words and len(extra) <= 2 and extra <= ALLOWED_EXTRA and (best is None or len(extra) < best[0]):
                best = (len(extra), key)
        if best is None:
            return None
        i = self.by_title[best[1]]
        return self._hit(i, 1.0, True), self.lead(i)

    def find_title(self, topic: str, min_ratio: float = 88, scorer=fuzz.ratio) -> int | None:
        """Exact (also singular) title match, else a close fuzzy match on the title (min_ratio > 100 disables the fuzzy step)."""
        for cand in (topic, topic[:-1] if topic.endswith("s") else topic, topic[:-2] if topic.endswith("es") else topic):
            if cand in self.by_title:
                return self.by_title[cand]
        if min_ratio > 100:
            return None
        hit = process.extractOne(topic, self.by_title.keys(), scorer=scorer, score_cutoff=min_ratio)
        return self.by_title[hit[0]] if hit else None

    def search(self, question: str) -> tuple[Hit | None, bool]:
        """Best article for the question and whether it was a definition-style question."""
        topic, is_def = topic_of(question)
        if is_def:
            i = self.find_title(topic, 101)
            if i is not None:
                return self._hit(i, 1.0, True), True
        i, score = self._nearest(topic if is_def else clean_query(question))
        return self._hit(i, score, False), is_def

    def confident_definition(self, hit: Hit, topic: str) -> bool:
        """Read the lead aloud only when the article really is the thing asked about."""
        return hit.title_match or (fuzz.ratio(topic, clean_query(hit.title)) >= 90 and hit.score >= 0.5)

    def answer(self, question: str) -> tuple[str, Hit | None, str]:
        """('extract', hit, text) to read aloud, ('context', hit, text) to help the LLM, or ('none', None, '')."""
        office = office_topic(question)
        if office:                                      # present-tense office question
            found = self.holder_sentence(office)
            if found:
                return "extract", found[0], f"According to Wikipedia as of {self.asof}: {found[1]}"
            i = self.find_title(office, 101)
            if i is not None:                           # the office exists but no biography states its holder: say so honestly
                hit = self._hit(i, 1.0, True)
                return "extract", hit, (f"I could not confirm who holds that office in my offline Wikipedia data from {self.asof}. "
                                        + hit.snippet(max_words=30, sentences=1))
        won = self.event_answer(question)
        if won:
            return "extract", won[0], f"According to Wikipedia as of {self.asof}: " + Hit(won[0].title, won[1], 1.0, True).snippet(max_words=60, sentences=3)
        hit, is_def = self.search(question)
        if hit is None:
            return "none", None, ""
        topic, _ = topic_of(question)
        if is_def and self.confident_definition(hit, topic):
            return "extract", hit, hit.snippet()
        # only trust a passage whose title is actually named in the question (a similar-sounding article misleads the LLM)
        named = fuzz.partial_ratio(clean_query(hit.title), clean_query(question)) >= 92 and len(hit.title) >= 4
        if named and hit.score >= self.rag_threshold:
            return "context", hit, hit.snippet(max_words=60, sentences=3)
        return "none", hit, ""
