import pytest
from multi_doc_chat.observability.alerting import (
    AlertManager,
    AlertEvent,
    AlertSeverity,
    AlertType,
)


def test_latency_anomaly_triggered_when_exceeding_threshold():
    """Verify alert is triggered and classified as WARNING or CRITICAL when latency exceeds threshold."""
    mgr = AlertManager(latency_threshold_ms=1000.0)

    # 1. Below threshold -> No alert
    normal = mgr.check_latency_anomaly(latency_ms=450.0, session_id="s1")
    assert normal is None
    assert len(mgr.alerts) == 0

    # 2. Exceeds threshold (< 2x) -> WARNING alert
    warn = mgr.check_latency_anomaly(latency_ms=1500.0, session_id="s1", run_id="r1")
    assert warn is not None
    assert warn.severity == AlertSeverity.WARNING
    assert warn.alert_type == AlertType.HIGH_LATENCY
    assert warn.metric_value == 1500.0
    assert len(mgr.alerts) == 1

    # 3. Exceeds threshold (>= 2x) -> CRITICAL alert
    crit = mgr.check_latency_anomaly(latency_ms=2500.0, session_id="s1", run_id="r2")
    assert crit is not None
    assert crit.severity == AlertSeverity.CRITICAL
    assert len(mgr.alerts) == 2


def test_feedback_anomaly_triggered_on_low_score():
    """Verify alert is triggered when user feedback score is below acceptable threshold."""
    mgr = AlertManager(low_score_threshold=0.60)

    # 1. High score -> No alert
    good = mgr.check_feedback_anomaly(score=1.0, run_id="r_good")
    assert good is None

    # 2. Moderate low score (0.4) -> WARNING alert
    warn = mgr.check_feedback_anomaly(score=0.4, run_id="r_warn", comment="Answer was partially incomplete")
    assert warn is not None
    assert warn.severity == AlertSeverity.WARNING
    assert warn.alert_type == AlertType.LOW_USER_SATISFACTION
    assert warn.metric_value == 0.4

    # 3. Very low score (0.0) -> CRITICAL alert
    crit = mgr.check_feedback_anomaly(score=0.0, run_id="r_crit", comment="Hallucinated information")
    assert crit is not None
    assert crit.severity == AlertSeverity.CRITICAL


def test_pipeline_error_alert_critical():
    """Verify pipeline errors trigger a CRITICAL alert."""
    mgr = AlertManager()
    alert = mgr.record_pipeline_error("Groq rate limit exceeded (429)", session_id="s_err", run_id="r_err")
    assert alert.severity == AlertSeverity.CRITICAL
    assert alert.alert_type == AlertType.PIPELINE_ERROR
    assert "Groq rate limit" in alert.message


def test_alert_manager_buffer_and_stats():
    """Verify in-memory buffer capacity limit and statistics aggregation."""
    mgr = AlertManager(latency_threshold_ms=500.0, max_buffered_alerts=3)

    mgr.check_latency_anomaly(latency_ms=600.0, session_id="s1")
    mgr.check_latency_anomaly(latency_ms=1200.0, session_id="s2")
    mgr.check_latency_anomaly(latency_ms=1500.0, session_id="s3")
    assert len(mgr.alerts) == 3

    # Add 4th -> oldest should be evicted
    mgr.check_latency_anomaly(latency_ms=2000.0, session_id="s4")
    assert len(mgr.alerts) == 3
    assert mgr.alerts[0].session_id == "s2"
    assert mgr.alerts[-1].session_id == "s4"

    stats = mgr.get_alert_statistics()
    assert stats["total_alerts"] == 3
    assert stats["critical_count"] > 0
    assert "HIGH_LATENCY" in stats["type_breakdown"]


def test_webhook_dispatch_called_when_url_provided(monkeypatch):
    """Verify webhook payload is dispatched when webhook_url is configured."""
    dispatched_payloads = []

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, json=None, **kwargs):
            dispatched_payloads.append((url, json))

    import httpx
    monkeypatch.setattr(httpx, "Client", MockClient)

    mgr = AlertManager(latency_threshold_ms=500.0, webhook_url="https://hooks.slack.com/test")
    mgr.check_latency_anomaly(latency_ms=900.0, session_id="s_webhook")

    assert len(dispatched_payloads) == 1
    url, payload = dispatched_payloads[0]
    assert url == "https://hooks.slack.com/test"
    assert "LLMOps Alert" in payload["text"]
    assert payload["alert"]["alert_type"] == "HIGH_LATENCY"
