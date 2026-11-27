import os
import time
import uuid
from enum import Enum
from typing import Optional, Dict, Any, List
import httpx
from pydantic import BaseModel, Field

from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.config_loader import load_config


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    HIGH_LATENCY = "HIGH_LATENCY"
    LOW_USER_SATISFACTION = "LOW_USER_SATISFACTION"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    CACHE_DEGRADATION = "CACHE_DEGRADATION"


class AlertEvent(BaseModel):
    alert_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: float = Field(default_factory=time.time)
    details: Dict[str, Any] = Field(default_factory=dict)


class AlertManager:
    """
    Real-time Anomaly Detection and Alerting for LLMOps.
    Monitors latency spikes, low user satisfaction, and system errors.
    Dispatches alerts to webhooks (Slack/Teams/PagerDuty) and stores in-memory history.
    """

    def __init__(
        self,
        latency_threshold_ms: float = 2500.0,
        low_score_threshold: float = 0.60,
        max_buffered_alerts: int = 100,
        webhook_url: Optional[str] = None,
    ):
        self.latency_threshold_ms = latency_threshold_ms
        self.low_score_threshold = low_score_threshold
        self.max_buffered_alerts = max_buffered_alerts
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")
        self.alerts: List[AlertEvent] = []
        self.redis_client = None
        self._init_redis()

    def _init_redis(self) -> None:
        target_url = os.getenv("REDIS_URL")
        if not target_url:
            try:
                from multi_doc_chat.utils.config_loader import load_config
                target_url = load_config().get("distributed_cache", {}).get("redis_url")
            except Exception:
                target_url = None
        target_url = target_url or "redis://127.0.0.1:6379/0"
        try:
            import redis
            client = redis.from_url(target_url, socket_timeout=1.0, decode_responses=True)
            client.ping()
            self.redis_client = client
            log.info("AlertManager connected to distributed Redis buffer", url=target_url)
        except Exception:
            self.redis_client = None

    @classmethod
    def from_config(cls) -> "AlertManager":
        """Instantiate AlertManager from project config.yaml."""
        try:
            cfg = load_config().get("alerting", {})
            return cls(
                latency_threshold_ms=float(cfg.get("latency_threshold_ms", 2500.0)),
                low_score_threshold=float(cfg.get("low_score_threshold", 0.60)),
                max_buffered_alerts=int(cfg.get("max_buffered_alerts", 100)),
            )
        except Exception:
            return cls()

    def _record_alert(self, alert: AlertEvent) -> None:
        """Buffer alert in local memory and shared Redis, and notify external webhooks if configured."""
        if len(self.alerts) >= self.max_buffered_alerts:
            self.alerts.pop(0)
        self.alerts.append(alert)

        if self.redis_client is not None:
            try:
                import json
                self.redis_client.lpush("rag:alerts:history", json.dumps(alert.model_dump()))
                self.redis_client.ltrim("rag:alerts:history", 0, self.max_buffered_alerts - 1)
            except Exception as re:
                log.debug("Redis alert buffer notice", error=str(re))

        log.warning(
            "AnomalyAlertTriggered",
            alert_id=alert.alert_id,
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
            message=alert.message,
            session_id=alert.session_id,
            run_id=alert.run_id,
        )

        # Dispatch async webhook if URL is set
        if self.webhook_url:
            self._send_webhook_sync(alert)

    def _send_webhook_sync(self, alert: AlertEvent) -> None:
        """Send notification payload to webhook."""
        try:
            payload = {
                "text": f"🚨 *LLMOps Alert: [{alert.severity.value}] {alert.alert_type.value}*\n{alert.message}",
                "alert": alert.model_dump(),
            }
            with httpx.Client(timeout=3.0) as client:
                client.post(self.webhook_url, json=payload)
        except Exception as e:
            log.error("Failed to dispatch alert webhook", error=str(e), url=self.webhook_url)

    def check_latency_anomaly(
        self,
        latency_ms: float,
        session_id: str,
        run_id: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Optional[AlertEvent]:
        """Trigger an alert if query latency exceeds operational threshold."""
        if latency_ms >= self.latency_threshold_ms:
            severity = AlertSeverity.CRITICAL if latency_ms >= (self.latency_threshold_ms * 2) else AlertSeverity.WARNING
            alert = AlertEvent(
                alert_type=AlertType.HIGH_LATENCY,
                severity=severity,
                message=f"High latency anomaly: {round(latency_ms, 2)}ms exceeded threshold of {self.latency_threshold_ms}ms.",
                session_id=session_id,
                run_id=run_id,
                metric_value=round(latency_ms, 2),
                threshold=self.latency_threshold_ms,
                details={"query": query[:120] if query else None},
            )
            self._record_alert(alert)
            return alert
        return None

    def check_feedback_anomaly(
        self,
        score: float,
        run_id: str,
        session_id: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> Optional[AlertEvent]:
        """Trigger an alert if user satisfaction feedback falls below acceptable threshold."""
        if score <= self.low_score_threshold:
            severity = AlertSeverity.CRITICAL if score <= 0.2 else AlertSeverity.WARNING
            alert = AlertEvent(
                alert_type=AlertType.LOW_USER_SATISFACTION,
                severity=severity,
                message=f"Low user satisfaction score ({score}) logged for run {run_id}.",
                session_id=session_id,
                run_id=run_id,
                metric_value=score,
                threshold=self.low_score_threshold,
                details={"comment": comment},
            )
            self._record_alert(alert)
            return alert
        return None

    def record_pipeline_error(
        self,
        error_msg: str,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> AlertEvent:
        """Trigger a critical alert on unhandled pipeline exceptions."""
        alert = AlertEvent(
            alert_type=AlertType.PIPELINE_ERROR,
            severity=AlertSeverity.CRITICAL,
            message=f"Pipeline error encountered: {error_msg}",
            session_id=session_id,
            run_id=run_id,
            details={"error": error_msg},
        )
        self._record_alert(alert)
        return alert

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent alerts from Redis or memory buffer."""
        if self.redis_client is not None:
            try:
                import json
                items = self.redis_client.lrange("rag:alerts:history", 0, limit - 1)
                if items:
                    return [json.loads(x) for x in items]
            except Exception as re:
                log.debug("Redis lrange notice", error=str(re))

        return [a.model_dump() for a in reversed(self.alerts[-limit:])]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Aggregate statistics on alert occurrences and severities."""
        alerts = self.get_recent_alerts(limit=self.max_buffered_alerts)
        total = len(alerts)
        critical_count = sum(1 for a in alerts if a.get("severity") == AlertSeverity.CRITICAL.value or a.get("severity") == AlertSeverity.CRITICAL)
        warning_count = sum(1 for a in alerts if a.get("severity") == AlertSeverity.WARNING.value or a.get("severity") == AlertSeverity.WARNING)

        type_breakdown: Dict[str, int] = {}
        for a in alerts:
            val = a.get("alert_type")
            if val:
                type_breakdown[val] = type_breakdown.get(val, 0) + 1

        return {
            "total_alerts": total,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "type_breakdown": type_breakdown,
            "webhook_configured": bool(self.webhook_url),
            "active_thresholds": {
                "latency_threshold_ms": self.latency_threshold_ms,
                "low_score_threshold": self.low_score_threshold,
            },
        }

    def clear_alerts(self) -> None:
        """Clear all buffered alerts in local memory and Redis."""
        self.alerts.clear()
        if self.redis_client is not None:
            try:
                self.redis_client.delete("rag:alerts:history")
            except Exception as re:
                log.debug("Redis alert delete notice", error=str(re))
        log.info("Alerts buffer cleared")

    def trigger_test_alert(self, alert_type: str = "HIGH_LATENCY") -> AlertEvent:
        """Manually trigger a test anomaly alert for monitoring verification."""
        if alert_type == "LOW_USER_SATISFACTION":
            alert = AlertEvent(
                alert_type=AlertType.LOW_USER_SATISFACTION,
                severity=AlertSeverity.WARNING,
                message="[TEST ALERT] Low satisfaction score (0.35) detected on test query.",
                metric_value=0.35,
                threshold=self.low_score_threshold,
                details={"test": True, "source": "Manual Trigger"},
            )
        elif alert_type == "PIPELINE_ERROR":
            alert = AlertEvent(
                alert_type=AlertType.PIPELINE_ERROR,
                severity=AlertSeverity.CRITICAL,
                message="[TEST ALERT] Simulated pipeline timeout exception.",
                details={"test": True, "source": "Manual Trigger"},
            )
        else:
            alert = AlertEvent(
                alert_type=AlertType.HIGH_LATENCY,
                severity=AlertSeverity.CRITICAL,
                message=f"[TEST ALERT] Latency anomaly spike: 3420.5ms exceeded threshold of {self.latency_threshold_ms}ms.",
                metric_value=3420.5,
                threshold=self.latency_threshold_ms,
                details={"test": True, "source": "Manual Trigger"},
            )

        self._record_alert(alert)
        return alert


# Global Alert Manager Singleton
ALERT_MANAGER = AlertManager.from_config()
