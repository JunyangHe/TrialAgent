from __future__ import annotations

import html
import json
import os
import re
import subprocess
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from trial_agent.config import _dbg
from trial_agent.models import ToolError
from trial_agent.tools.base import HTTPConfig, HTTPToolAdapter, ToolAdapter, ToolResult


class BioMCPAdapter(ToolAdapter):
    """BioMCP adapter using the biomcp-python CLI.

    Uses subprocess to invoke biomcp commands (biomcp trial search/get):
      - discover: biomcp trial search -c "<query>" -j --page-size N
      - fetch: biomcp trial get <nct_id> -j

    Requires: pip install biomcp-python

    Environment variables:
      - BIOMCP_CMD (default: biomcp) – path to biomcp binary
      - BIOMCP_TIMEOUT (default: 60) – subprocess timeout in seconds
      - NCI_API_KEY (optional) – for --source nci
    """

    name = "biomcp"

    def __init__(self) -> None:
        self.cmd = os.getenv("BIOMCP_CMD", "biomcp")
        self.timeout = float(os.getenv("BIOMCP_TIMEOUT", "60"))

    def discover(self, query: str, filters: dict[str, Any]) -> ToolResult:
        limit = 25
        try:
            limit = min(int(filters.get("limit", 25)), 1000)
        except (TypeError, ValueError):
            pass
        args = [
            self.cmd,
            "trial",
            "search",
            "-c",
            query,
            "-j",
            "--page-size",
            str(limit),
        ]
        if filters.get("status"):
            status = filters["status"]
            if isinstance(status, list):
                status = status[0] if status else None
            if status:
                s = str(status).lower()
                if s in ("open", "closed", "any"):
                    args.extend(["-s", s])
                elif "recruit" in s or "open" in s:
                    args.extend(["-s", "open"])
        if filters.get("phase"):
            phase = filters["phase"]
            if isinstance(phase, list):
                phase = phase[0] if phase else None
            if phase:
                p = str(phase).lower().replace("phase", "").strip()
                if p in ("early_phase1", "phase1", "phase2", "phase3", "phase4", "not_applicable"):
                    args.extend(["-p", p])
                elif p in ("1", "2", "3", "4"):
                    args.extend(["-p", f"phase{p}"])

        if filters.get("interventions"):
            interv = filters["interventions"]
            if isinstance(interv, list):
                interv = interv[0] if interv else None
            if interv:
                args.extend(["-i", str(interv)])
        geographies = filters.get("geographies")
        if geographies:
            geo = geographies[0] if isinstance(geographies, list) and geographies else geographies
            if geo and str(geo).strip():
                args.extend(["--facility", str(geo).strip()])

        _dbg(f"biomcp discover: args={args}")
        raw = self._run(args)
        records = self._extract_records(raw)
        _dbg(f"biomcp discover: raw keys={list(raw.keys()) if isinstance(raw, dict) else 'not-dict'}, extracted {len(records)} records")
        if records and isinstance(raw, dict):
            _dbg(f"biomcp discover: first raw record keys={list(records[0].keys())[:15]}")
        return ToolResult(
            records=[self._normalize_record(row) for row in records],
            metadata={"query": query, "filters": filters},
        )

    def fetch(self, trial_id: str) -> ToolResult:
        nct = trial_id.upper()
        if not nct.startswith("NCT"):
            nct = f"NCT{nct}" if nct.isdigit() else trial_id
        args = [self.cmd, "trial", "get", nct, "-j"]
        _dbg(f"biomcp fetch: args={args}")
        raw = self._run(args)
        row = self._extract_single_record(raw)
        _dbg(f"biomcp fetch: raw keys={list(raw.keys()) if isinstance(raw, dict) else 'not-dict'}, extracted row={row is not None}")
        if row is None:
            return ToolResult(records=[], metadata={"trial_id": trial_id})
        return ToolResult(
            records=[self._normalize_record(row)],
            metadata={"trial_id": trial_id},
        )

    def _run(self, args: list[str]) -> dict[str, Any]:
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ},
            )
        except FileNotFoundError as exc:
            raise ToolError(
                f"biomcp not found. Install from https://github.com/genomoncology/biomcp "
                "(curl -fsSL .../install.sh | bash)"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"biomcp timed out after {self.timeout}s for: {' '.join(args)}") from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[:500]
            _dbg(f"biomcp _run: returncode={result.returncode}, stderr={stderr[:200]!r}, stdout len={len(result.stdout or '')}")
            raise ToolError(f"biomcp failed ({result.returncode}): {stderr or result.stdout}")

        if not result.stdout.strip():
            return {}
        try:
            out = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            _dbg(f"biomcp _run: JSON decode error, stdout preview={(result.stdout or '')[:200]!r}")
            raise ToolError(f"biomcp returned invalid JSON: {result.stdout[:300]}") from exc
        _dbg(f"biomcp _run: success, stdout len={len(result.stdout or '')}, parsed type={type(out).__name__}")
        return out if isinstance(out, dict) else {"data": out}

    @staticmethod
    def _extract_records(response: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("records", "trials", "results", "data", "items"):
            value = response.get(key)
            if isinstance(value, list):
                return [v for v in value if isinstance(v, dict)]
            if isinstance(value, dict):
                nested = value.get("records") or value.get("results")
                if isinstance(nested, list):
                    return [v for v in nested if isinstance(v, dict)]
        return []

    @classmethod
    def _extract_single_record(cls, response: dict[str, Any]) -> dict[str, Any] | None:
        direct = response.get("record") or response.get("trial")
        if isinstance(direct, dict):
            return direct
        records = cls._extract_records(response)
        if records:
            return records[0]
        # biomcp "trial get" returns ClinicalTrials.gov API v2 format (protocolSection, etc)
        if isinstance(response, dict) and (
            "protocolSection" in response or "nctId" in str(response)
        ):
            return cls._flatten_ctgov_v2_response(response)
        return None

    @staticmethod
    def _extract_locations_from_protocol(prot: dict[str, Any]) -> list[str]:
        """Extract location strings from protocolSection.contactsLocationsModule."""
        locs_mod = prot.get("contactsLocationsModule") if isinstance(prot, dict) else {}
        locs = locs_mod.get("locations") if isinstance(locs_mod, dict) else []
        if not isinstance(locs, list):
            return []
        result: list[str] = []
        for loc in locs:
            if not isinstance(loc, dict):
                continue
            parts = [loc.get("facility"), loc.get("city"), loc.get("state"), loc.get("country")]
            s = ", ".join(str(p).strip() for p in parts if p and str(p).strip())
            if s:
                result.append(s)
        return result

    @classmethod
    def _flatten_ctgov_v2_response(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Flatten ClinicalTrials.gov API v2 / biomcp trial get response."""
        prot = raw.get("protocolSection") or {}
        ident = prot.get("identificationModule") or {}
        status_mod = prot.get("statusModule") or {}
        sponsor_mod = prot.get("sponsorCollaboratorsModule") or {}
        desc = prot.get("descriptionModule") or {}
        conditions_mod = prot.get("conditionsModule") or {}
        interventions_mod = prot.get("interventionsModule") or {}

        nct_id = ident.get("nctId") or raw.get("nctId")
        brief = ident.get("briefTitle") or ident.get("officialTitle") or ""
        lead = sponsor_mod.get("leadSponsor") or {}
        sponsor_name = lead.get("name") if isinstance(lead, dict) else None
        cond_list = conditions_mod.get("conditions") or []
        interv_list = interventions_mod.get("interventions") or []
        if isinstance(interv_list, list):
            interv_list = [
                x.get("name") or x.get("interventionName") if isinstance(x, dict) else str(x)
                for x in interv_list
            ]

        start_struct = status_mod.get("startDateStruct") or {}
        prime_struct = status_mod.get("primaryCompletionDateStruct") or status_mod.get("completionDateStruct") or {}
        start_date = start_struct.get("date") if isinstance(start_struct, dict) else None
        completion = prime_struct.get("date") if isinstance(prime_struct, dict) else None

        return {
            "nct_id": nct_id,
            "nctId": nct_id,
            "trial_key": nct_id,
            "title": brief,
            "brief_title": brief,
            "conditions": cond_list if isinstance(cond_list, list) else [cond_list],
            "interventions": interv_list,
            "sponsor": sponsor_name,
            "lead_sponsor": sponsor_name,
            "status": status_mod.get("overallStatus"),
            "phase": None,
            "study_type": None,
            "start_date": start_date,
            "primary_completion_date": completion,
            "Completion Date": completion,
            "summary": desc.get("briefSummary"),
            "brief_summary": desc.get("briefSummary"),
            "locations": cls._extract_locations_from_protocol(prot),
        }

    @staticmethod
    def _normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
        nct_id = (
            raw.get("nct_id")
            or raw.get("nctId")
            or raw.get("nctNumber")
            or raw.get("NCT Number")
        )
        trial_key = raw.get("trial_key") or raw.get("id") or raw.get("registry_id") or nct_id
        conditions = raw.get("conditions") or raw.get("condition") or raw.get("Conditions") or []
        interventions = raw.get("interventions") or raw.get("intervention") or raw.get("Interventions") or []

        if isinstance(conditions, str):
            conditions = [s.strip() for s in conditions.split("|") if s.strip()] if conditions else []
        if isinstance(interventions, str):
            interventions = [s.strip() for s in interventions.split("|") if s.strip()] if interventions else []

        title = (
            raw.get("title")
            or raw.get("brief_title")
            or raw.get("briefTitle")
            or raw.get("Study Title")
            or ""
        )
        sponsor = (
            raw.get("sponsor")
            or raw.get("lead_sponsor")
            or raw.get("leadSponsor")
            or raw.get("Sponsor")
        )
        status = (
            raw.get("status")
            or raw.get("overall_status")
            or raw.get("overallStatus")
            or raw.get("Study Status")
        )
        phase = raw.get("phase") or raw.get("Phases")
        study_type = raw.get("study_type") or raw.get("studyType") or raw.get("Study Type")
        start_date = raw.get("start_date") or raw.get("startDate") or raw.get("Start Date")
        completion = (
            raw.get("primary_completion_date")
            or raw.get("primaryCompletionDate")
            or raw.get("Completion Date")
        )
        summary = (
            raw.get("summary")
            or raw.get("brief_summary")
            or raw.get("briefSummary")
            or raw.get("Brief Summary")
        )
        locations = raw.get("locations") or raw.get("Locations") or []

        return {
            "trial_key": trial_key,
            "id": raw.get("id", trial_key),
            "nct_id": nct_id,
            "title": title,
            "conditions": conditions,
            "interventions": interventions,
            "sponsor": sponsor,
            "status": status,
            "phase": phase,
            "study_type": study_type,
            "start_date": start_date,
            "primary_completion_date": completion,
            "summary": summary,
            "locations": [str(x) for x in locations] if isinstance(locations, list) else [str(locations)],
            "outcomes": raw.get("outcomes") or {},
            "source": "biomcp",
            "identifiers": {
                "primary": trial_key,
                **({"nct": nct_id} if nct_id else {}),
                **({"registry": raw["registry_id"]} if raw.get("registry_id") else {}),
            },
        }


class ClinicalTrialsGovV2Adapter(HTTPToolAdapter):
    name = "ctgov_v2"

    def __init__(self) -> None:
        super().__init__(
            HTTPConfig(
                base_url=os.getenv("CTGOV_V2_BASE_URL", "https://clinicaltrials.gov/api/v2"),
                timeout_s=float(os.getenv("CTGOV_V2_TIMEOUT", "20")),
            )
        )

    def discover(self, query: str, filters: dict[str, Any]) -> ToolResult:
        try:
            page_size = int(filters.get("limit", 25))
        except (TypeError, ValueError):
            page_size = 25
        page_size = max(1, min(page_size, 1000))

        params = {"query.term": query, "pageSize": page_size, "format": "json"}
        if filters.get("pageToken"):
            params["pageToken"] = str(filters["pageToken"])
        geographies = filters.get("geographies")
        if geographies and isinstance(geographies, list) and geographies:
            params["query.locn"] = " ".join(str(g).strip() for g in geographies if str(g).strip())
        elif geographies and isinstance(geographies, str) and geographies.strip():
            params["query.locn"] = str(geographies).strip()

        raw = self._request_json("GET", "/studies", params=params)
        studies = raw.get("studies")
        if not isinstance(studies, list):
            studies = []

        records = [self._normalize_study(study) for study in studies if isinstance(study, dict)]
        records = [record for record in records if record.get("trial_key")]
        return ToolResult(
            records=records,
            metadata={
                "query": query,
                "filters": filters,
                "nextPageToken": raw.get("nextPageToken"),
                "totalCount": raw.get("totalCount"),
            },
        )

    def fetch(self, trial_id: str) -> ToolResult:
        nct_id = self._normalize_nct_id(trial_id)
        raw = self._request_json("GET", f"/studies/{quote(nct_id)}", params={"format": "json"}, allow_404=True)
        if not raw:
            return ToolResult(records=[], metadata={"trial_id": trial_id})

        record = self._normalize_study(raw)
        if not record.get("trial_key"):
            record["trial_key"] = nct_id
            record["id"] = nct_id
            record["identifiers"] = {"primary": nct_id, "nct": nct_id}
        return ToolResult(records=[record], metadata={"trial_id": trial_id})

    @staticmethod
    def _normalize_nct_id(trial_id: str) -> str:
        value = (trial_id or "").strip().upper()
        if value.startswith("NCT"):
            return value
        return f"NCT{value}" if value.isdigit() else value

    @staticmethod
    def _to_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []

    @staticmethod
    def _extract_date(value: Any) -> str | None:
        if isinstance(value, dict):
            date_value = value.get("date")
            return str(date_value) if date_value else None
        if isinstance(value, str) and value:
            return value
        return None

    @staticmethod
    def _extract_locations(protocol: dict[str, Any]) -> list[str]:
        """Extract location strings from protocolSection.contactsLocationsModule."""
        locs_mod = protocol.get("contactsLocationsModule") if isinstance(protocol, dict) else {}
        locs = locs_mod.get("locations") if isinstance(locs_mod, dict) else []
        if not isinstance(locs, list):
            return []
        result: list[str] = []
        for loc in locs:
            if not isinstance(loc, dict):
                continue
            parts = [loc.get("facility"), loc.get("city"), loc.get("state"), loc.get("country")]
            s = ", ".join(str(p).strip() for p in parts if p and str(p).strip())
            if s:
                result.append(s)
        return result

    @classmethod
    def _normalize_study(cls, raw: dict[str, Any]) -> dict[str, Any]:
        protocol = raw.get("protocolSection") if isinstance(raw.get("protocolSection"), dict) else {}
        ident = protocol.get("identificationModule") if isinstance(protocol.get("identificationModule"), dict) else {}
        status_mod = protocol.get("statusModule") if isinstance(protocol.get("statusModule"), dict) else {}
        design_mod = protocol.get("designModule") if isinstance(protocol.get("designModule"), dict) else {}
        cond_mod = protocol.get("conditionsModule") if isinstance(protocol.get("conditionsModule"), dict) else {}
        arms_mod = (
            protocol.get("armsInterventionsModule")
            if isinstance(protocol.get("armsInterventionsModule"), dict)
            else {}
        )
        sponsor_mod = (
            protocol.get("sponsorCollaboratorsModule")
            if isinstance(protocol.get("sponsorCollaboratorsModule"), dict)
            else {}
        )
        desc_mod = protocol.get("descriptionModule") if isinstance(protocol.get("descriptionModule"), dict) else {}
        outcomes_mod = protocol.get("outcomesModule") if isinstance(protocol.get("outcomesModule"), dict) else {}

        nct_id = ident.get("nctId")
        trial_key = raw.get("trial_key") or raw.get("id") or nct_id

        interventions: list[str] = []
        for intervention in arms_mod.get("interventions", []):
            if not isinstance(intervention, dict):
                continue
            name = intervention.get("name")
            if isinstance(name, str) and name.strip():
                interventions.append(name.strip())

        phases = cls._to_str_list(design_mod.get("phases"))
        phase = ", ".join(phases) if phases else None

        lead_sponsor = sponsor_mod.get("leadSponsor")
        sponsor: str | None = None
        if isinstance(lead_sponsor, dict):
            name = lead_sponsor.get("name")
            sponsor = str(name) if name else None
        elif isinstance(lead_sponsor, str):
            sponsor = lead_sponsor

        return {
            "trial_key": trial_key,
            "id": trial_key,
            "nct_id": nct_id,
            "title": ident.get("briefTitle") or ident.get("officialTitle") or "",
            "conditions": cls._to_str_list(cond_mod.get("conditions")),
            "interventions": interventions,
            "sponsor": sponsor,
            "status": status_mod.get("overallStatus"),
            "phase": phase,
            "study_type": design_mod.get("studyType"),
            "start_date": cls._extract_date(status_mod.get("startDateStruct")),
            "primary_completion_date": cls._extract_date(status_mod.get("primaryCompletionDateStruct")),
            "summary": desc_mod.get("briefSummary") or desc_mod.get("detailedDescription"),
            "locations": cls._extract_locations(protocol),
            "outcomes": {
                "primaryOutcomes": outcomes_mod.get("primaryOutcomes", []),
                "secondaryOutcomes": outcomes_mod.get("secondaryOutcomes", []),
            },
            "source": "ctgov_v2",
            "identifiers": {
                "primary": trial_key,
                **({"nct": nct_id} if nct_id else {}),
            },
        }


class WHOICTRPAdapter(HTTPToolAdapter):
    name = "who_ictrp"

    def __init__(self) -> None:
        super().__init__(
            HTTPConfig(
                base_url=os.getenv("WHO_ICTRP_BASE_URL", "https://trialsearch.who.int"),
                timeout_s=float(os.getenv("WHO_ICTRP_TIMEOUT", "20")),
                default_headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "User-Agent": "trialagent/0.1 (+https://trialsearch.who.int)",
                },
            )
        )
        self._fallback_bases = [self.http.base_url, "https://apps.who.int/trialsearch"]

    def discover(self, query: str, filters: dict[str, Any]) -> ToolResult:
        limit = self._bounded_limit(filters.get("limit", 25), upper=1000)
        query = (query or "").strip()
        geographies = filters.get("geographies")
        if geographies and isinstance(geographies, list) and geographies:
            query = f"{query} {' '.join(str(g).strip() for g in geographies if str(g).strip())}".strip()
        elif geographies and isinstance(geographies, str) and geographies.strip():
            query = f"{query} {geographies.strip()}".strip()
        if not query:
            return ToolResult(records=[], metadata={"query": query, "filters": filters})

        direct_trial_id = self._extract_trial_id(query)
        if direct_trial_id:
            fetched = self.fetch(direct_trial_id)
            if fetched.records:
                return ToolResult(records=fetched.records[:limit], metadata={"query": query, "filters": filters})

        candidate_pages: list[tuple[str, str, dict[str, Any]]] = [
            ("GET", "/AdvSearch.aspx", {"SearchTerm": query}),
            ("GET", "/AdvSearch.aspx", {"SearchTermSimple": query}),
            ("GET", "/", {"SearchTerm": query}),
        ]

        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        used_base = ""
        for base_url in self._fallback_bases:
            for method, path, params in candidate_pages:
                page = self._request_text_on_base(base_url, method, path, params=params, allow_404=True)
                if not page:
                    continue
                trial_refs = self._extract_trial_refs(page)
                for trial_id, title in trial_refs:
                    if trial_id in seen:
                        continue
                    seen.add(trial_id)
                    used_base = base_url
                    collected.append(self._build_stub_record(trial_id, title=title))
                    if len(collected) >= limit:
                        break
                if len(collected) >= limit:
                    break
            if len(collected) >= limit:
                break

        return ToolResult(
            records=collected,
            metadata={"query": query, "filters": filters, "source_base": used_base or self.http.base_url},
        )

    def fetch(self, trial_id: str) -> ToolResult:
        normalized_id = (trial_id or "").strip()
        if not normalized_id:
            return ToolResult(records=[], metadata={"trial_id": trial_id})

        for base_url in self._fallback_bases:
            page = self._request_text_on_base(
                base_url,
                "GET",
                "/Trial2.aspx",
                params={"TrialID": normalized_id},
                allow_404=True,
            )
            if not page:
                continue
            page_lower = page.lower()
            if "trialid" not in page_lower and "public title" not in page_lower and "scientific title" not in page_lower:
                continue
            record = self._normalize_trial_page(page, normalized_id)
            return ToolResult(records=[record], metadata={"trial_id": trial_id, "source_base": base_url})

        return ToolResult(records=[], metadata={"trial_id": trial_id})

    @staticmethod
    def _bounded_limit(value: Any, *, upper: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 25
        return max(1, min(parsed, upper))

    @staticmethod
    def _strip_tags(fragment: str) -> str:
        text = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
        text = re.sub(r"(?i)</(p|div|li|tr|td|th|h[1-6])>", "\n", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(unquote(text))
        return re.sub(r"[ \t\r\f\v]+", " ", text).strip()

    @classmethod
    def _extract_trial_id(cls, value: str) -> str | None:
        text = (value or "").strip()
        match = re.search(r"\b([A-Z]{2,}[A-Z0-9\-]*\d{2,})\b", text.upper())
        if match:
            return match.group(1)
        return None

    @classmethod
    def _extract_trial_refs(cls, page: str) -> list[tuple[str, str]]:
        refs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for href, label in re.findall(r"<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", page, flags=re.I | re.S):
            parsed = urlparse(html.unescape(href))
            query = parse_qs(parsed.query)
            trial_ids = query.get("TrialID") or query.get("trialid")
            if not trial_ids:
                # Also allow direct Trial2.aspx?TrialID=... patterns in malformed URLs.
                match = re.search(r"trialid=([^&\"'>]+)", href, flags=re.I)
                if not match:
                    continue
                trial_id = unquote(match.group(1)).strip()
            else:
                trial_id = unquote(str(trial_ids[0])).strip()
            if not trial_id or trial_id in seen:
                continue
            seen.add(trial_id)
            title = cls._strip_tags(label)
            refs.append((trial_id, title))
        return refs

    def _request_text_on_base(
        self,
        base_url: str,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> str:
        original = self.http.base_url
        try:
            self.http.base_url = base_url
            return self._request_text(method, path, params=params, allow_404=allow_404)
        except ToolError:
            return ""
        finally:
            self.http.base_url = original

    @staticmethod
    def _split_items(value: str | None) -> list[str]:
        if not value:
            return []
        normalized = re.sub(r"[;\n\r]+", "|", value)
        items = [item.strip(" -\t") for item in normalized.split("|")]
        return [item for item in items if item]

    @classmethod
    def _extract_labeled_values(cls, page: str) -> dict[str, str]:
        fields: dict[str, str] = {}

        # Common table layout: label cell followed by value cell.
        for raw_label, raw_value in re.findall(
            r"<tr[^>]*>\s*<t[dh][^>]*>(.*?)</t[dh]>\s*<t[dh][^>]*>(.*?)</t[dh]>\s*</tr>",
            page,
            flags=re.I | re.S,
        ):
            label = cls._normalize_label(cls._strip_tags(raw_label))
            value = cls._strip_tags(raw_value)
            if label and value and label not in fields:
                fields[label] = value

        # Fallback for text blocks with "Label: Value".
        plain = cls._strip_tags(page)
        for line in plain.splitlines():
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            label = cls._normalize_label(left)
            value = right.strip()
            if label and value and label not in fields:
                fields[label] = value
        return fields

    @staticmethod
    def _normalize_label(label: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()

    @classmethod
    def _pick_field(cls, fields: dict[str, str], *needles: str) -> str | None:
        for key, value in fields.items():
            for needle in needles:
                if needle in key:
                    return value
        return None

    @classmethod
    def _build_stub_record(cls, trial_id: str, *, title: str | None = None) -> dict[str, Any]:
        nct_id = trial_id if trial_id.upper().startswith("NCT") else None
        clean_title = (title or "").strip() or trial_id
        return {
            "trial_key": trial_id,
            "id": trial_id,
            "nct_id": nct_id,
            "title": clean_title,
            "conditions": [],
            "interventions": [],
            "sponsor": None,
            "status": None,
            "phase": None,
            "study_type": None,
            "start_date": None,
            "primary_completion_date": None,
            "summary": None,
            "locations": [],
            "outcomes": {},
            "source": "who_ictrp",
            "identifiers": {"primary": trial_id, **({"nct": nct_id} if nct_id else {})},
        }

    @classmethod
    def _normalize_trial_page(cls, page: str, trial_id: str) -> dict[str, Any]:
        fields = cls._extract_labeled_values(page)
        record = cls._build_stub_record(trial_id)

        title = cls._pick_field(fields, "public title", "scientific title", "title")
        conditions = cls._pick_field(fields, "condition", "problem studied", "health condition")
        interventions = cls._pick_field(fields, "intervention")
        sponsor = cls._pick_field(fields, "primary sponsor", "sponsor")
        status = cls._pick_field(fields, "recruitment status", "status")
        phase = cls._pick_field(fields, "phase")
        study_type = cls._pick_field(fields, "study type", "study design")
        start_date = cls._pick_field(fields, "date of first enrolment", "date of first enrollment", "start date")
        completion = cls._pick_field(fields, "completion date", "date of completion")
        summary = cls._pick_field(fields, "brief summary", "summary", "objective", "description")
        primary_outcome = cls._pick_field(fields, "primary outcome")
        secondary_outcome = cls._pick_field(fields, "secondary outcome")
        country = cls._pick_field(fields, "country", "countries", "recruitment country", "countries of recruitment")
        location = cls._pick_field(fields, "location", "study location", "sites")
        locations_list: list[str] = []
        for val in (country, location):
            if val:
                locations_list.extend(cls._split_items(val))

        if title:
            record["title"] = title
        record["conditions"] = cls._split_items(conditions)
        record["interventions"] = cls._split_items(interventions)
        record["sponsor"] = sponsor
        record["status"] = status
        record["phase"] = phase
        record["study_type"] = study_type
        record["start_date"] = start_date
        record["primary_completion_date"] = completion
        record["summary"] = summary
        record["outcomes"] = {
            "primaryOutcomes": cls._split_items(primary_outcome),
            "secondaryOutcomes": cls._split_items(secondary_outcome),
        }
        if locations_list:
            record["locations"] = locations_list

        # If the page exposes a registry ID that differs from the requested ID, keep it as auxiliary metadata.
        secondary_id = cls._pick_field(fields, "secondary id", "trial id")
        if secondary_id and secondary_id != trial_id:
            record["identifiers"]["secondary"] = secondary_id
        return record


def build_tool_registry() -> dict[str, ToolAdapter]:
    return {
        "biomcp": BioMCPAdapter(),
        "ctgov_v2": ClinicalTrialsGovV2Adapter(),
        "who_ictrp": WHOICTRPAdapter(),
    }
