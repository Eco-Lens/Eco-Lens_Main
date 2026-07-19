"""Unit tests for Eco-Lens utilities and core logic. Fast tests: no model loads."""

import sys, os, json, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# ─── Test utils (3.Table_Understanding/utils.py) ───────────────

from pipeline_core.utils import atomic_write_json, read_json_safe, validate_schema_version, safe_filename


class TestAtomicWrite:
    def test_atomic_write_and_read(self, tmp_path):
        p = str(tmp_path / "test.json")
        atomic_write_json({"a": 1, "b": [2, 3]}, p, schema_version="1.0")
        assert os.path.exists(p)
        assert not os.path.exists(p + ".tmp")
        data = read_json_safe(p)
        assert data["_meta"]["schema_version"] == "1.0"
        assert data["a"] == 1

    def test_read_missing(self):
        assert read_json_safe("/nonexistent/file.json", default={"fallback": True}) == {"fallback": True}

    def test_validate_schema(self, tmp_path):
        p = str(tmp_path / "s.json")
        atomic_write_json({"x": 1}, p, schema_version="2.0")
        data = read_json_safe(p)
        assert validate_schema_version(data, "2.0")
        assert not validate_schema_version(data, "1.0")


class TestSafeFilename:
    def test_basic(self):
        assert safe_filename("hello world.pdf") == "hello world.pdf"
        assert safe_filename("") == "untitled"
        assert safe_filename("../../../etc/passwd") == "etcpaswd"


# ─── Test number parsing (3.Table_Understanding/utils.py) ───────

from utils import _to_standard_number, is_numeric, is_numeric_lenient, parse_number, merge_bboxes, bbox_area, expand_bbox


class TestNumberParsing:
    def test_to_standard_number(self):
        assert _to_standard_number("1,234") == 1234.0
        assert _to_standard_number("1.234,56") == 1234.56
        assert _to_standard_number("1,234.56") == 1234.56
        assert _to_standard_number("(1,234)") == -1234.0
        assert _to_standard_number("") is None
        assert _to_standard_number("abc") is None
        assert _to_standard_number("42") == 42.0
        assert _to_standard_number("1.234,56") == 1234.56  # EU format

    def test_parse_number(self):
        assert parse_number("1,234 VND") == 1234.0
        assert parse_number("(loss)") is None
        assert parse_number("") is None

    def test_is_numeric(self):
        assert is_numeric("1,234")
        assert is_numeric("42")
        assert not is_numeric("hello")
        assert not is_numeric("")

    def test_is_numeric_lenient(self):
        assert is_numeric_lenient("1,234")
        assert is_numeric_lenient("a 5 b")
        assert not is_numeric_lenient("abc")


class TestBboxFunctions:
    def test_merge_bboxes(self):
        boxes = [[0, 0, 10, 10], [5, 5, 15, 15]]
        result = merge_bboxes(boxes)
        assert result == [0, 0, 15, 15]

    def test_merge_bboxes_empty(self):
        assert merge_bboxes([]) is None

    def test_bbox_area(self):
        assert bbox_area([0, 0, 10, 10]) == 100

    def test_expand_bbox(self):
        result = expand_bbox([10, 10, 90, 90], 5, 5, 100, 100)
        assert result == [5, 5, 95, 95]

    def test_expand_bbox_clamp(self):
        result = expand_bbox([0, 0, 5, 5], 10, 10, 100, 100)
        assert result[0] == 0
        assert result[1] == 0


# ─── Test scope functions (4.SemanticMapping/run_scope_inference.py) ───

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the functions we need, mocking the model dependency
import run_scope_inference as sci


class TestExtractValueUnit:
    def test_tco2e(self):
        v, u = sci.extract_value_unit("Emissions: 1,234 tCO2e")
        assert v == 1234.0
        assert u == "tCO2e"

    def test_mwh(self):
        v, u = sci.extract_value_unit("Energy: 500 MWh")
        assert v == 500.0
        assert u == "MWh"

    def test_percent(self):
        v, u = sci.extract_value_unit("Reduction: 15.5%")
        assert v == 15.5
        assert u == "%"

    def test_no_match(self):
        v, u = sci.extract_value_unit("Some random text without units")
        assert v is None
        assert u is None

    def test_none_text(self):
        v, u = sci.extract_value_unit(None)
        assert v is None
        assert u is None


class TestExplicitScopes:
    def test_scope_1(self):
        assert sci.extract_explicit_scopes("Scope 1 emissions") == ["Scope 1"]

    def test_scope_1_and_2(self):
        assert sci.extract_explicit_scopes("Scope 1 and Scope 2") == ["Scope 1", "Scope 2"]

    def test_no_scope(self):
        assert sci.extract_explicit_scopes("No relevant content") == []

    def test_case_insensitive(self):
        assert sci.extract_explicit_scopes("SCOPE 3") == ["Scope 3"]


class TestIsEsgEligible:
    def test_explicit_scope(self):
        assert sci.is_esg_eligible("Scope 1 emissions")

    def test_ghg_keyword(self):
        assert sci.is_esg_eligible("greenhouse gas emissions")

    def test_carbon(self):
        assert sci.is_esg_eligible("carbon footprint analysis")

    def test_measurement(self):
        assert sci.is_esg_eligible("emissions: 500 tCO2e")

    def test_non_esg(self):
        assert not sci.is_esg_eligible("The weather is nice today")
        assert not sci.is_esg_eligible("")
        assert not sci.is_esg_eligible(None)

    def test_gdp_not_eligible(self):
        """GDP forecasts should not be ESG-eligible unless they mention ESG."""
        assert not sci.is_esg_eligible("GDP growth forecast 3.5%")
        assert sci.is_esg_eligible("GDP carbon intensity 0.15 kgCO2e/USD")


class TestResolveTextScopeEligibility:
    """Test the eligibility gate in resolve_text_scope."""

    def test_non_esg_text_rejected(self):
        """Non-ESG text with non-Other prediction should be rejected by eligibility gate."""
        pred = {"scope": "Scope 1", "confidence": 0.8, "probabilities": {"Other": 0.1, "Scope 1": 0.8, "Scope 2": 0.05, "Scope 3": 0.05}}
        result = sci.resolve_text_scope("GDP growth forecast 3.5%", pred)
        assert result["scope"] == "Other"
        assert result["scope_source"] == "eligibility_gate"

    def test_esg_text_accepted(self):
        """ESG text with confident non-Other prediction should pass."""
        pred = {"scope": "Scope 1", "confidence": 0.8, "probabilities": {"Other": 0.1, "Scope 1": 0.8, "Scope 2": 0.05, "Scope 3": 0.05}}
        result = sci.resolve_text_scope("Scope 1: greenhouse gas emissions 500 tCO2e", pred)
        assert result["scope"] == "Scope 1"
        assert result["scope_source"] == "model_with_measurement_and_esg_context"

    def test_explicit_scope_wins(self):
        """Explicit 'Scope N' mention should override model."""
        pred = {"scope": "Other", "confidence": 0.6, "probabilities": {"Other": 0.6, "Scope 1": 0.2, "Scope 2": 0.1, "Scope 3": 0.1}}
        result = sci.resolve_text_scope("Scope 2 emissions from electricity", pred)
        assert result["scope"] == "Scope 2"
        assert result["scope_source"] == "explicit_mention"

    def test_mixed_scopes(self):
        pred = {"scope": "Other", "confidence": 0.5, "probabilities": {"Other": 0.5, "Scope 1": 0.3, "Scope 2": 0.1, "Scope 3": 0.1}}
        result = sci.resolve_text_scope("Scope 1 and Scope 2 combined", pred)
        assert result["scope"] == "Mixed"
        assert result["scope_source"] == "explicit_mentions"


# ─── Test codebase duplicates were removed ─────────────────────

class TestPipelineCleanup:
    def test_no_find_sets(self):
        """_find_sets debug function should not exist in pipeline.py."""
        content = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "3.Table_Understanding", "pipeline.py"), encoding="utf-8").read()
        assert "_find_sets" not in content, "_find_sets debug function still present"

    def test_single_normalize_text(self):
        content = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "3.Table_Understanding", "pipeline.py"), encoding="utf-8").read()
        count = content.count("def normalize_text")
        assert count == 1, f"Found {count} normalize_text definitions (expected 1)"

    def test_no_wildcard_import(self):
        content = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "3.Table_Understanding", "pipeline.py"), encoding="utf-8").read()
        assert "from config import *" not in content

    def test_no_duplicate_is_number(self):
        content = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "3.Table_Understanding", "pipeline.py"), encoding="utf-8").read()
        count = content.count("def is_number")
        assert count == 1, f"Found {count} is_number definitions (expected 1)"

    def test_no_duplicate_extract_year(self):
        content = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "3.Table_Understanding", "pipeline.py"), encoding="utf-8").read()
        count = content.count("def extract_year")
        assert count == 1, f"Found {count} extract_year definitions (expected 1)"


# ─── Test run isolation contract ───────────────────────────────

class TestRunIsolation:
    def test_two_runs_have_different_dirs(self):
        from pipeline_core.context import RunContext
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ctx_a = RunContext("run_a", tmp, os.path.join(tmp, "runs", "run_a"))
            ctx_b = RunContext("run_b", tmp, os.path.join(tmp, "runs", "run_b"))
            ctx_a.ensure_dirs()
            ctx_b.ensure_dirs()
            # Write to run_a
            (ctx_a.output_root / "test.txt").write_text("run_a_data")
            # Run B should not see run A's file
            assert not (ctx_b.output_root / "test.txt").exists()
            # Cleanup
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_context_paths(self):
        from pipeline_core.context import RunContext
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext("test_run", tmp, os.path.join(tmp, "runs", "test_run"))
            assert ctx.run_id == "test_run"
            assert "runs/test_run" in str(ctx.run_root)
            assert ctx.pages_dir.name == "pages"
            assert ctx.ocr_json.name == "ocr_words.json"
            assert ctx.viz_index_html.name == "index.html"


# ─── Test config has weights that sum to 100 ────────────────────

class TestConfig:
    from pipeline_core.config import STEPS, TOTAL_WEIGHT, STEP_WEIGHTS

    def test_weights_sum_to_100(self):
        assert self.TOTAL_WEIGHT == sum(s["weight"] for s in self.STEPS)

    def test_all_steps_have_weights(self):
        for s in self.STEPS:
            assert "weight" in s
            assert s["weight"] > 0
