package scaler

import (
	"fmt"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type Metrics struct {
	healthy          *prometheus.GaugeVec
	queueDepthApprox *prometheus.GaugeVec
	reg              prometheus.Gatherer
}

func NewMetrics(reg prometheus.Registerer) *Metrics {
	healthy := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "token_scaler_healthy",
		Help: "1 if the scaler phase is healthy, 0 on error.",
	}, []string{"phase"})
	queueDepth := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "token_scaler_queue_depth_approx",
		Help: "Approximate message depth from queue broker (drift telemetry only).",
	}, []string{"queue"})
	reg.MustRegister(healthy, queueDepth)
	return &Metrics{
		healthy:          healthy,
		queueDepthApprox: queueDepth,
		reg:              reg.(prometheus.Gatherer),
	}
}

func (m *Metrics) SetHealthy(phase string, healthy bool) {
	v := 0.0
	if healthy {
		v = 1.0
	}
	m.healthy.WithLabelValues(phase).Set(v)
}

func (m *Metrics) RecordQueueDrift(queueName string, approxDepth int64) {
	m.queueDepthApprox.WithLabelValues(queueName).Set(float64(approxDepth))
}

func (m *Metrics) BuildHTTPMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "ok")
	})
	mux.Handle("/metrics", promhttp.HandlerFor(m.reg, promhttp.HandlerOpts{}))
	return mux
}

func (m *Metrics) StartHTTPServer(port string) error {
	return http.ListenAndServe(":"+port, m.BuildHTTPMux())
}
