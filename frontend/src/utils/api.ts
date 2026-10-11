import axios from 'axios'
import type {
  SystemHealth, SystemStats,
  ObservationListResponse, Observation,
  EventListResponse, Event,
  CorrelationResult, CorrelationDecision,
  Dataset, ModelRecord,
  TrainingJob, EvaluationJob,
  InferenceResult, RadarProcessResult,
  ReplayState, SyntheticRunResult,
  RadarTrack,
} from '../types'

const BASE = '/api'

const api = axios.create({ baseURL: BASE })

// ─── System ──────────────────────────────────────────────────────────────────
export const getHealth = () => api.get<SystemHealth>('/health').then(r => r.data)
export const getStats = () => api.get<SystemStats>('/system/stats').then(r => r.data)
export const getLogs = (level?: string, limit = 50) =>
  api.get('/system/logs', { params: { level, limit } }).then(r => r.data)

// ─── Datasets ────────────────────────────────────────────────────────────────
export const listDatasets = () => api.get<Dataset[]>('/datasets').then(r => r.data)
export const validateDataset = (id: string) =>
  api.post(`/datasets/validate/${id}`).then(r => r.data)
export const updateDatasetPath = (id: string, localPath: string) =>
  api.patch(`/datasets/${id}/path`, null, { params: { local_path: localPath } }).then(r => r.data)

// ─── Models ──────────────────────────────────────────────────────────────────
export const listModels = () => api.get<ModelRecord[]>('/models').then(r => r.data)
export const getModel = (id: string) => api.get<ModelRecord>(`/models/${id}`).then(r => r.data)
export const getModelLoadStatus = (id: string) =>
  api.get(`/models/${id}/load-status`).then(r => r.data)

// ─── Inference ───────────────────────────────────────────────────────────────
export const runSatelliteInference = (
  file: File,
  confidence = 0.25,
  iou = 0.45,
  modelId?: string,
) => {
  const form = new FormData()
  form.append('file', file)
  return api.post<InferenceResult>('/inference/satellite', form, {
    params: { confidence, iou, model_id: modelId },
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

export const runThermalInference = (
  file: File,
  confidence = 0.25,
  iou = 0.45,
  modelId?: string,
) => {
  const form = new FormData()
  form.append('file', file)
  return api.post<InferenceResult>('/inference/thermal', form, {
    params: { confidence, iou, model_id: modelId },
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

export const runRadarImport = (file: File) => {
  const form = new FormData()
  form.append('file', file)
  return api.post<RadarProcessResult>('/inference/radar', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

// ─── Observations ─────────────────────────────────────────────────────────────
export const listObservations = (sensorType?: string, limit = 50, offset = 0) =>
  api.get<ObservationListResponse>('/observations', {
    params: { sensor_type: sensorType, limit, offset },
  }).then(r => r.data)

// ─── Correlation ─────────────────────────────────────────────────────────────
export const runCorrelation = (
  observationIds: string[] = [],
  temporalWindow?: number,
  spatialThreshold?: number,
) =>
  api.post<CorrelationResult>(
    '/correlation/run',
    observationIds,
    { params: { temporal_window: temporalWindow, spatial_threshold: spatialThreshold } },
  ).then(r => r.data)

export const getCorrelationHistory = (limit = 50) =>
  api.get('/correlation/history', { params: { limit } }).then(r => r.data)

// ─── Events ──────────────────────────────────────────────────────────────────
export const listEvents = (priority?: string, status?: string, limit = 50, offset = 0) =>
  api.get<EventListResponse>('/events', { params: { priority, status, limit, offset } }).then(r => r.data)

export const getEvent = (id: string) => api.get<Event>(`/events/${id}`).then(r => r.data)

export const autoGenerateEvents = (observationIds: string[] = []) =>
  api.post('/events/auto-generate', observationIds).then(r => r.data)

export const updateEventReview = (id: string, reviewStatus: string, notes?: string) =>
  api.patch(`/events/${id}/review`, { review_status: reviewStatus, notes }).then(r => r.data)

// ─── Training ────────────────────────────────────────────────────────────────
export const startTraining = (payload: {
  model_id: string
  dataset_name: string
  epochs: number
  batch_size: number
  image_size: number
  learning_rate?: number
  device: string
}) => api.post('/training/start', payload).then(r => r.data)

export const listTrainingJobs = () =>
  api.get<TrainingJob[]>('/training/jobs').then(r => r.data)

export const getTrainingJob = (id: string) =>
  api.get<TrainingJob>(`/training/jobs/${id}`).then(r => r.data)

export const cancelTrainingJob = (id: string) =>
  api.post(`/training/cancel/${id}`).then(r => r.data)

// ─── Evaluation ──────────────────────────────────────────────────────────────
export const startEvaluation = (modelId: string, datasetName: string, split = 'val') =>
  api.post('/evaluation/start', { model_id: modelId, dataset_name: datasetName, dataset_split: split }).then(r => r.data)

export const listEvaluationJobs = () =>
  api.get<EvaluationJob[]>('/evaluation/jobs').then(r => r.data)

// ─── Radar ───────────────────────────────────────────────────────────────────
export const getRadarTracks = () =>
  api.get<{ tracks: RadarTrack[]; status: unknown }>('/radar/tracks').then(r => r.data)

export const getActiveRadarTracks = () =>
  api.get<{ tracks: RadarTrack[]; active_count: number }>('/radar/tracks/active').then(r => r.data)

export const resetRadarTracks = () =>
  api.delete('/radar/tracks/reset').then(r => r.data)

// ─── Replay ──────────────────────────────────────────────────────────────────
export const loadReplayFile = (file: File) => {
  const form = new FormData()
  form.append('file', file)
  return api.post<ReplayState>('/replay/load', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

export const controlReplay = (action: string, speed?: number) =>
  api.post('/replay/control', { action, speed }).then(r => r.data)

export const getReplayStatus = () =>
  api.get<ReplayState>('/replay/status').then(r => r.data)

export const getReplayProcessed = (limit = 50, offset = 0) =>
  api.get('/replay/processed', { params: { limit, offset } }).then(r => r.data)

// ─── Synthetic ───────────────────────────────────────────────────────────────
export const runSyntheticScenarios = () =>
  api.post<SyntheticRunResult>('/synthetic/run').then(r => r.data)

// ─── Reports ─────────────────────────────────────────────────────────────────
export const listReports = () => api.get('/reports').then(r => r.data)
export const exportReport = (id: string) => api.get(`/reports/export/${id}`).then(r => r.data)
