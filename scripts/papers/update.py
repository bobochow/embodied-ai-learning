#!/usr/bin/env python3
"""Build a review-first daily embodied-AI paper radar update.

The pipeline treats external survey repositories as discovery sources only. It
normalizes their arXiv identifiers, fetches canonical metadata from arXiv,
applies transparent relevance scoring, records Top 10 -> Top 3 -> Top 1 traces,
and prepends a date section to the existing VitePress page.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "paper-radar.json"
LATEST_PATH = ROOT / "docs" / "papers" / "latest.md"
INDEX_PATH = ROOT / "data" / "papers" / "index.json"
RUNS_DIR = ROOT / "data" / "papers" / "runs"
RUN_TIMEZONE = ZoneInfo("Asia/Shanghai")

ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
ATOM = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    sources: list[str] = field(default_factory=list)
    source_sections: list[str] = field(default_factory=list)
    track: str = "vla"
    priority: str = "P1"
    relevance_score: int = 0
    score_reasons: list[str] = field(default_factory=list)
    summary_zh: str = ""
    signals: list[str] = field(default_factory=list)


def canonical_arxiv_id(value: str) -> str | None:
    match = re.search(r"(?:abs/|pdf/)?(\d{4}\.\d{4,5})(?:v\d+)?", value, re.I)
    return match.group(1) if match else None


def request_text(url: str, *, headers: dict[str, str] | None = None, retries: int = 3) -> str:
    request_headers = {"User-Agent": "embodied-ai-paper-radar/1.0"}
    request_headers.update(headers or {})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {url}: {last_error}")


def extract_ids_by_section(markdown: str, sections: Iterable[str], limit: int) -> dict[str, list[str]]:
    wanted = {section.casefold(): section for section in sections}
    results: dict[str, list[str]] = {section: [] for section in sections}
    current: str | None = None
    for line in markdown.splitlines():
        heading = SECTION_RE.match(line)
        if heading:
            current = wanted.get(heading.group(1).strip().casefold())
            continue
        if not current or len(results[current]) >= limit:
            continue
        for raw_id in ARXIV_ID_RE.findall(line):
            paper_id = canonical_arxiv_id(raw_id)
            if paper_id and paper_id not in results[current]:
                results[current].append(paper_id)
    return results


def fetch_source_candidates(config: dict[str, Any]) -> tuple[dict[str, dict[str, set[str]]], dict[str, Any]]:
    candidates: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"sources": set(), "sections": set()}
    )
    provenance: dict[str, Any] = {}
    limit = int(config["max_per_source_section"])
    for source_name, source in config["sources"].items():
        markdown = request_text(source["readme_url"])
        by_section = extract_ids_by_section(markdown, source["sections"], limit)
        for section, ids in by_section.items():
            for paper_id in ids:
                candidates[paper_id]["sources"].add(source_name)
                candidates[paper_id]["sections"].add(section)
        provenance[source_name] = {
            "repository": source["repository"],
            "readme_url": source["readme_url"],
            "readme_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            "sections": {name: len(ids) for name, ids in by_section.items()},
        }
    return candidates, provenance


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _text(node: ET.Element | None) -> str:
    return " ".join((node.text if node is not None and node.text else "").split())


def fetch_arxiv_metadata(candidate_ids: list[str]) -> dict[str, Paper]:
    papers: dict[str, Paper] = {}
    for batch in chunks(candidate_ids, 40):
        query = urllib.parse.urlencode({"id_list": ",".join(batch), "max_results": len(batch)})
        xml_text = request_text(f"https://export.arxiv.org/api/query?{query}", retries=4)
        root = ET.fromstring(xml_text)
        for entry in root.findall("atom:entry", ATOM):
            raw_id = _text(entry.find("atom:id", ATOM))
            paper_id = canonical_arxiv_id(raw_id)
            if not paper_id:
                continue
            papers[paper_id] = Paper(
                arxiv_id=paper_id,
                title=_text(entry.find("atom:title", ATOM)),
                abstract=_text(entry.find("atom:summary", ATOM)),
                authors=[_text(author.find("atom:name", ATOM)) for author in entry.findall("atom:author", ATOM)],
                categories=[node.attrib.get("term", "") for node in entry.findall("atom:category", ATOM)],
                published=_text(entry.find("atom:published", ATOM)),
                updated=_text(entry.find("atom:updated", ATOM)),
            )
        time.sleep(1)
    return papers


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value[:10])


def classify_and_score(paper: Paper, negative_terms: list[str]) -> Paper:
    text = f"{paper.title} {paper.abstract}".casefold()
    title = paper.title.casefold()
    sections = " ".join(paper.source_sections).casefold()
    score = 0
    reasons: list[str] = []

    strong_terms = (
        "vision-language-action",
        "vision language action",
        "world-action model",
        "world action model",
        "robot policy",
        "robot manipulation",
        "robot learning",
        "humanoid robot",
        "latent action",
        "human-to-robot",
    )
    robot_terms = ("robot", "manipulation", "visuomotor", "humanoid", "embodied")
    action_terms = ("action", "policy", "control", "trajectory", "demonstration")

    if any(term in text for term in strong_terms):
        score += 4
        reasons.append("strong embodied-action phrase")
    if any(term in text for term in robot_terms) and any(term in text for term in action_terms):
        score += 2
        reasons.append("robot and action context")
    if "cs.RO" in paper.categories:
        score += 2
        reasons.append("cs.RO category")
    if "awesome_vla_wam" in paper.sources:
        score += 3
        reasons.append("curated awesome-vla-wam source")
    if "daily_arxiv" in paper.sources:
        score += 1
        reasons.append("daily arXiv source")
    if any(term.casefold() in text for term in negative_terms) and "robot" not in text:
        score -= 6
        reasons.append("generic non-robot world-model domain")

    if any(term in text for term in ("tactile", "haptic", "force-aware", "visuo-tactile")):
        paper.track = "tactile"
    elif "humanoid" in text:
        paper.track = "humanoid"
    elif any(term in text for term in ("world-action model", "world action model", "video-action")):
        paper.track = "wam"
    elif any(term in title for term in ("vision-language-action", "vision language action", "vla")):
        paper.track = "vla"
    elif any(term in title or term in sections for term in (
        "dataset",
        "benchmark",
        "human action pretraining",
        "human-to-robot",
        "latent action",
    )):
        paper.track = "data"
    else:
        paper.track = "vla"

    if any(term in text for term in ("real-world", "real robot", "unitree", "ur5", "franka")):
        score += 1
        reasons.append("real-robot evidence in abstract")
    if any(term in text for term in ("code is available", "project page", "github.com")):
        score += 1
        reasons.append("resource link in abstract")

    paper.relevance_score = score
    paper.score_reasons = reasons
    paper.priority = "P0" if score >= 10 else "P1"
    paper.signals = infer_signals(paper)
    return paper


def infer_signals(paper: Paper) -> list[str]:
    text = f"{paper.title} {paper.abstract}".casefold()
    labels: list[str] = []
    by_term = [
        (("vision-language-action", "vision language action"), "VLA"),
        (("world-action", "world action"), "WAM"),
        (("human-to-robot", "human video", "egocentric"), "HUMAN-TO-ROBOT"),
        (("latent action",), "LATENT ACTION"),
        (("humanoid",), "HUMANOID"),
        (("tactile", "haptic"), "TACTILE"),
        (("benchmark",), "BENCHMARK"),
        (("reinforcement learning", "policy optimization"), "RL"),
        (("real-world", "real robot"), "REAL ROBOT"),
        (("open-source", "code is available"), "OPEN SOURCE"),
    ]
    for terms, label in by_term:
        if any(term in text for term in terms) and label not in labels:
            labels.append(label)
    fallback = {
        "vla": "ROBOT POLICY",
        "wam": "WORLD MODEL",
        "data": "DATA/EVAL",
        "humanoid": "HUMANOID",
        "tactile": "TACTILE",
    }
    if not labels:
        labels.append(fallback[paper.track])
    return labels[:3]


def recent_enough(paper: Paper, cutoff: dt.date) -> bool:
    return max(parse_date(paper.published), parse_date(paper.updated)) >= cutoff


def select_papers(papers: list[Paper], max_publish: int) -> tuple[list[Paper], dict[str, Any]]:
    grouped: dict[str, list[Paper]] = defaultdict(list)
    for paper in papers:
        grouped[paper.track].append(paper)
    trace: dict[str, Any] = {}
    top_three_pool: list[Paper] = []
    for track in ("vla", "wam", "data", "humanoid", "tactile"):
        ranked = sorted(
            grouped.get(track, []),
            key=lambda paper: (paper.relevance_score, paper.updated, paper.arxiv_id),
            reverse=True,
        )
        top_ten = ranked[:10]
        top_three = top_ten[:3]
        top_one = top_three[:1]
        trace[track] = {
            "top_10": [paper.arxiv_id for paper in top_ten],
            "top_3": [paper.arxiv_id for paper in top_three],
            "top_1": [paper.arxiv_id for paper in top_one],
        }
        top_three_pool.extend(top_three)

    selected: list[Paper] = []
    seen: set[str] = set()
    for paper in sorted(
        top_three_pool,
        key=lambda item: (item.relevance_score, item.updated, item.arxiv_id),
        reverse=True,
    ):
        if paper.arxiv_id not in seen and len(selected) < max_publish:
            selected.append(paper)
            seen.add(paper.arxiv_id)

    if len(selected) < max_publish:
        for paper in sorted(
            papers,
            key=lambda item: (item.relevance_score, item.updated, item.arxiv_id),
            reverse=True,
        ):
            if paper.arxiv_id not in seen:
                selected.append(paper)
                seen.add(paper.arxiv_id)
            if len(selected) >= max_publish:
                break
    return selected, trace


def _extract_json_array(value: str) -> list[dict[str, Any]]:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.S)
    data = json.loads(value)
    if not isinstance(data, list):
        raise ValueError("LLM response must be a JSON array")
    return data


def enrich_with_anthropic(papers: list[Paper]) -> bool:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return False
    model = os.environ.get("PAPER_LLM_MODEL", "claude-sonnet-4-6")
    payload_papers = [
        {"arxiv_id": paper.arxiv_id, "title": paper.title, "abstract": paper.abstract}
        for paper in papers
    ]
    prompt = (
        "你是具身智能论文编辑。仅依据给定标题和摘要，为每篇论文生成80到150字中文摘要，"
        "不得补充输入中没有的机构、项目页、代码或实验数字。返回严格JSON数组，每项仅含"
        "arxiv_id、summary_zh、signals；signals是1到3个简短大写英文标签。\n\n"
        + json.dumps(payload_papers, ensure_ascii=False)
    )
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 5000,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "user-agent": "embodied-ai-paper-radar/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = "".join(block.get("text", "") for block in result.get("content", []))
        enriched = {item["arxiv_id"]: item for item in _extract_json_array(content)}
        for paper in papers:
            item = enriched.get(paper.arxiv_id)
            if not item:
                continue
            paper.summary_zh = str(item.get("summary_zh", "")).strip()
            signals = [sanitize_signal(value) for value in item.get("signals", [])]
            paper.signals = [value for value in signals if value][:3] or paper.signals
        return all(paper.summary_zh for paper in papers)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[paper-radar] Anthropic enrichment failed; using review fallback: {exc}", file=sys.stderr)
        return False


def sanitize_signal(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9+./ -]", "", str(value).upper()).strip()
    return re.sub(r"\s+", " ", cleaned)[:32]


def fallback_summary(paper: Paper) -> str:
    abstract = paper.abstract.strip()
    if len(abstract) > 420:
        abstract = abstract[:417].rsplit(" ", 1)[0] + "..."
    return f"自动候选，待中文细读：{abstract}"


def render_article(paper: Paper, *, featured: bool) -> str:
    classes = ["paper-ticket", f"paper-ticket--{paper.track}"]
    if featured:
        classes.append("paper-ticket--featured")
    meta: list[str] = []
    if featured:
        meta.append('<span class="paper-editor-pick">EDITOR PICK</span>')
    meta.extend(f"<span>{html.escape(signal)}</span>" for signal in paper.signals)
    meta.append(f"<span>{paper.priority}</span>")
    meta.append('<span class="paper-status paper-status--todo">待细读</span>')
    summary = paper.summary_zh or fallback_summary(paper)
    return (
        f'<article class="{" ".join(classes)}">'
        f'<div class="paper-ticket__meta">{"".join(meta)}</div>'
        f'<h3><a href="https://arxiv.org/abs/{paper.arxiv_id}" target="_blank" rel="noreferrer">'
        f'{html.escape(paper.title)}</a></h3>'
        f'<p>{html.escape(summary)}</p>'
        f'<div class="paper-ticket__links"><a href="https://arxiv.org/abs/{paper.arxiv_id}" '
        f'target="_blank" rel="noreferrer">arXiv</a></div></article>'
    )


def render_day_section(run_date: dt.date, papers: list[Paper], trace: dict[str, Any]) -> str:
    featured_ids = {
        values["top_1"][0]
        for values in trace.values()
        if values.get("top_1")
    }
    names = " × ".join(paper.title.split(":", 1)[0] for paper in papers[:6])
    articles = "\n    ".join(
        render_article(paper, featured=paper.arxiv_id in featured_ids)
        for paper in papers
    )
    date = run_date.isoformat()
    return (
        f'<h2 id="papers-{date}" class="paper-day-heading">{date}</h2>\n\n'
        '<div class="daily-paper-section">\n'
        f'  <p class="paper-day-note"><strong>自动候选 · 待人工复核</strong>优先检查 {html.escape(names)}。'
        '候选来自 DailyArxiv 与 awesome-vla-wam，并以 arXiv 元数据重新核验。</p>\n'
        '  <div class="paper-queue-grid">\n'
        f'    {articles}\n'
        '  </div>\n'
        '</div>'
    )


def update_latest_document(original: str, run_date: dt.date, papers: list[Paper], trace: dict[str, Any]) -> str:
    date = run_date.isoformat()
    if f'id="papers-{date}"' in original:
        raise ValueError(f"paper section already exists for {date}")
    first_heading = re.search(r'<h2 id="papers-\d{4}-\d{2}-\d{2}"', original)
    if not first_heading:
        raise ValueError("could not locate the first dated paper section")

    section = render_day_section(run_date, papers, trace)
    updated = original[: first_heading.start()] + section + "\n\n" + original[first_heading.start() :]
    p0_count = sum(paper.priority == "P0" for paper in papers)
    total_count = len(re.findall(r'<article class="[^"]*paper-ticket', updated))

    replacements = [
        (r'<time datetime="\d{4}-\d{2}-\d{2}">[^<]+</time>', f'<time datetime="{date}">{date.replace("-", ".")}</time>'),
        (r'<p class="paper-brief__dek">.*?</p>', f'<p class="paper-brief__dek">今日自动筛出 {len(papers)} 篇强相关候选，来自每日 arXiv 检索与人工策展 VLA/WAM 清单，等待编辑复核。</p>'),
        (r'(<dd data-paper-stat="latest">)\d+(</dd>)', rf'\g<1>{len(papers)}\g<2>'),
        (r'(<dd data-paper-stat="p0">)\d+(</dd>)', rf'\g<1>{p0_count}\g<2>'),
        (r'(<dd data-paper-stat="done">)\d+(</dd>)', r'\g<1>0\g<2>'),
        (r'(<dd data-paper-stat="date">)[^<]+(</dd>)', rf'\g<1>{run_date:%m.%d}\g<2>'),
        (r'(data-paper-filter-count[^>]*>显示 )\d+( 篇</output>)', rf'\g<1>{total_count}\g<2>'),
    ]
    for pattern, replacement in replacements:
        updated, count = re.subn(pattern, replacement, updated, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f"expected one hero replacement for pattern: {pattern}")
    return updated


def existing_arxiv_ids(markdown: str) -> set[str]:
    return {match.group(1) for match in ARXIV_ID_RE.finditer(markdown)}


def has_date_section(markdown: str, run_date: dt.date) -> bool:
    return f'id="papers-{run_date.isoformat()}"' in markdown


def load_index(markdown: str) -> dict[str, Any]:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return {
        "version": 1,
        "papers": {
            paper_id: {"first_seen": "historical", "sources": ["site_archive"]}
            for paper_id in sorted(existing_arxiv_ids(markdown))
        },
    }


def validate_document(markdown: str) -> None:
    headings = list(re.finditer(r'<h2 id="papers-(\d{4}-\d{2}-\d{2})"', markdown))
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        section = markdown[heading.start() : end]
        articles = re.findall(
            r'<article class="[^"]*paper-ticket[^"]*">([\s\S]*?)</article>',
            section,
        )
        ids = []
        for article in articles:
            match = ARXIV_ID_RE.search(article)
            if match:
                ids.append(match.group(1))
        duplicates = sorted({paper_id for paper_id in ids if ids.count(paper_id) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate arXiv IDs in {heading.group(1)}: {', '.join(duplicates[:10])}"
            )
    hero_date = re.search(r'<time datetime="(\d{4}-\d{2}-\d{2})">', markdown)
    heading_date = re.search(r'<h2 id="papers-(\d{4}-\d{2}-\d{2})"', markdown)
    if not hero_date or not heading_date or hero_date.group(1) != heading_date.group(1):
        raise ValueError("hero date does not match first paper section")
    displayed = re.search(r'data-paper-filter-count[^>]*>显示 (\d+) 篇', markdown)
    articles = len(re.findall(r'<article class="[^"]*paper-ticket', markdown))
    if not displayed or int(displayed.group(1)) != articles:
        raise ValueError(f"displayed paper count does not match article count ({articles})")


def current_run_date(now: dt.datetime | None = None) -> dt.date:
    """Return the calendar day used by the Beijing-time workflow schedule."""
    instant = now or dt.datetime.now(dt.timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("current time must be timezone-aware")
    return instant.astimezone(RUN_TIMEZONE).date()


def run_update(args: argparse.Namespace) -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    markdown = LATEST_PATH.read_text(encoding="utf-8")
    if args.check:
        validate_document(markdown)
        print("[paper-radar] document validation passed")
        return 0

    run_date = dt.date.fromisoformat(args.today) if args.today else current_run_date()
    if has_date_section(markdown, run_date):
        print(f"[paper-radar] section already exists for {run_date.isoformat()}; nothing to do")
        return 0
    lookback_days = args.lookback_days or int(config["lookback_days"])
    cutoff = run_date - dt.timedelta(days=lookback_days)
    index = load_index(markdown)
    known_ids = set(index["papers"]) | existing_arxiv_ids(markdown)

    candidates, provenance = fetch_source_candidates(config)
    unknown_ids = sorted(set(candidates) - known_ids)
    if not unknown_ids:
        print("[paper-radar] no unseen source candidates")
        return 0
    metadata = fetch_arxiv_metadata(unknown_ids)
    eligible: list[Paper] = []
    rejected: dict[str, str] = {}
    for paper_id in unknown_ids:
        paper = metadata.get(paper_id)
        if not paper:
            rejected[paper_id] = "arXiv metadata unavailable"
            continue
        paper.sources = sorted(candidates[paper_id]["sources"])
        paper.source_sections = sorted(candidates[paper_id]["sections"])
        classify_and_score(paper, config["negative_terms"])
        if not recent_enough(paper, cutoff):
            rejected[paper_id] = "outside lookback window"
        elif paper.relevance_score < int(config["min_relevance_score"]):
            rejected[paper_id] = f"relevance score {paper.relevance_score} below threshold"
        else:
            eligible.append(paper)

    selected, trace = select_papers(eligible, args.max_publish or int(config["max_publish"]))
    if not selected:
        print("[paper-radar] no eligible unseen papers")
        return 0
    llm_enriched = enrich_with_anthropic(selected)
    updated = update_latest_document(markdown, run_date, selected, trace)
    validate_document(updated)

    report = {
        "run_date": run_date.isoformat(),
        "cutoff_date": cutoff.isoformat(),
        "llm_enriched": llm_enriched,
        "provenance": provenance,
        "candidate_count": len(unknown_ids),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "selection_trace": trace,
        "selected": [asdict(paper) for paper in selected],
        "rejected": rejected,
    }
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    LATEST_PATH.write_text(updated, encoding="utf-8")
    for paper in selected:
        index["papers"][paper.arxiv_id] = {
            "first_seen": run_date.isoformat(),
            "sources": paper.sources,
            "title": paper.title,
        }
    index["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{run_date.isoformat()}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[paper-radar] added {len(selected)} papers for {run_date.isoformat()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate generated files without network access")
    parser.add_argument("--dry-run", action="store_true", help="fetch and rank without writing files")
    parser.add_argument("--today", help="override run date (YYYY-MM-DD), primarily for reproducible tests")
    parser.add_argument("--lookback-days", type=int, help="override configured lookback window")
    parser.add_argument("--max-publish", type=int, help="override maximum published candidates")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run_update(build_parser().parse_args()))
    except (RuntimeError, ValueError, ET.ParseError, json.JSONDecodeError) as exc:
        print(f"[paper-radar] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
