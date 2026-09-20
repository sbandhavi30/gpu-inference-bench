{{- define "token-scaler.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "token-scaler.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- include "token-scaler.fullname" . }}
{{- else }}
default
{{- end }}
{{- end }}
