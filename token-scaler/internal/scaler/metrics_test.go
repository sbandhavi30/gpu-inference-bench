package scaler_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"token-scaler/internal/scaler"
)

func TestMetrics_SetHealthy_RecordsGauge(t *testing.T) {
	reg := prometheus.NewRegistry()
	m := scaler.NewMetrics(reg)
	m.SetHealthy("is_active", false)

	families, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, f := range families {
		if f.GetName() == "token_scaler_healthy" {
			for _, metric := range f.GetMetric() {
				for _, label := range metric.GetLabel() {
					if label.GetName() == "phase" && label.GetValue() == "is_active" {
						if metric.GetGauge().GetValue() == 0 {
							found = true
						}
					}
				}
			}
		}
	}
	if !found {
		t.Fatal("expected token_scaler_healthy{phase=is_active}=0")
	}
}

func TestMetrics_SetHealthy_True(t *testing.T) {
	reg := prometheus.NewRegistry()
	m := scaler.NewMetrics(reg)
	m.SetHealthy("get_metrics", true)

	families, _ := reg.Gather()
	for _, f := range families {
		if f.GetName() == "token_scaler_healthy" {
			for _, metric := range f.GetMetric() {
				for _, label := range metric.GetLabel() {
					if label.GetValue() == "get_metrics" {
						if metric.GetGauge().GetValue() != 1 {
							t.Fatalf("want 1, got %v", metric.GetGauge().GetValue())
						}
					}
				}
			}
		}
	}
}

func TestMetrics_RecordQueueDrift(t *testing.T) {
	reg := prometheus.NewRegistry()
	m := scaler.NewMetrics(reg)
	m.RecordQueueDrift("test-queue", 42)

	families, _ := reg.Gather()
	var found bool
	for _, f := range families {
		if f.GetName() == "token_scaler_queue_depth_approx" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected token_scaler_queue_depth_approx metric")
	}
}

func TestMetrics_HealthzEndpoint(t *testing.T) {
	reg := prometheus.NewRegistry()
	m := scaler.NewMetrics(reg)
	mux := m.BuildHTTPMux()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/healthz", nil)
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", w.Code)
	}
	body, _ := io.ReadAll(w.Body)
	if !strings.Contains(string(body), "ok") {
		t.Fatalf("want 'ok' body, got %q", body)
	}
}
