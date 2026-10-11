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
export interface ImageInferenceOptions {
  confidence?: number
  iou?: number
  modelId?: string
  tileSize?: number
  capturedAt?: string
  latitude?: number
  longitude?: number
  positionUncertaintyM?: number
}

const imageInference = (sensor: 'satellite' | 'thermal', file: File, o: ImageInferenceOptions) => {
  const form = new FormData()
  form.append('file', file)
  return api.post<InferenceResult>(`/inference/${sensor}`, form, {
    params: {
      confidence: o.confidence ?? 0.25, iou: o.iou ?? 0.45, model_id: o.modelId || undefined,
      tile_size: o.tileSize || undefined, captured_at: o.capturedAt || undefined,
      latitude: o.latitude, longitude: o.longitude, position_uncertainty_m: o.positionUncertaintyM,
    },
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

export const runSatelliteInference = (file: File, o: ImageInferenceOptions = {}) => imageInference('satellite', file, o)
export const runThermalInference = (file: File, o: ImageInferenceOptions = {}) => imageInference('thermal', file, o)

export const runRadarImport = (file: File, opts: { associationThreshold?: number; maxMisses?: number; reset?: boolean } = {}) => {
  const form = new FormData()
  form.append('file', file)
  return api.post<RadarProcessResult>('/inference/radar', form, {
    params: { association_threshold: opts.associationThreshold, max_misses: opts.maxMisses, reset: opts.reset ?? true },
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

// ─── Observations ─────────────────────────────────────────────────────────────
export const listObservations = (sensorType?: string, limit = 50, offset = 0, provenance?: string) =>
  api.get<ObservationListResponse>('/observations', {
    params: { sensor_type: sensorType || undefined, limit, offset, provenance: provenance || undefined },
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

export const autoGenerateEvents = (observationIds: string[] = [], includeSingletons = false) =>
  api.post('/events/auto-generate', observationIds, { params: { include_singletons: includeSingletons } }).then(r => r.data)

export const exportEvent = (id: string) => api.get(`/events/${id}/export`).then(r => r.data)

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

export const trainingPreflight = (modelId: string, datasetName: string, device: string) =>
  api.get('/training/preflight', { params: { model_id: modelId, dataset_name: datasetName, device } }).then(r => r.data)

// ─── Evaluation ──────────────────────────────────────────────────────────────
export const startEvaluation = (modelId: string, datasetName: string, split = 'val', maxImages?: number) =>
  api.post('/evaluation/start', { model_id: modelId, dataset_name: datasetName, dataset_split: split },
    { params: { max_images: maxImages || undefined } }).then(r => r.data)

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
export const listReports = (reportType?: string) =>
  api.get('/reports', { params: { report_type: reportType || undefined } }).then(r => r.data)
export const exportReport = (id: string) => api.get(`/reports/export/${id}`).then(r => r.data)
export const createFusionReport = () => api.post('/reports/fusion-tests').then(r => r.data)

// ─── Datasets (extra) ────────────────────────────────────────────────────────
export const inspectDataset = (id: string) => api.get(`/datasets/${id}/inspect`).then(r => r.data)
export const convertDataset = (id: string, sourcePath: string, tileSize = 640, maxImages?: number) =>
  api.post(`/datasets/${id}/convert`, null, { params: { source_path: sourcePath, tile_size: tileSize, max_images: maxImages || undefined } }).then(r => r.data)
