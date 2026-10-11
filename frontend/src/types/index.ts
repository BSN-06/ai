// ─── System ──────────────────────────────────────────────────────────────────
export interface SystemHealth {
  status: string
  app_name: string
  version: string
  database: string
  models_available: number
  datasets_registered: number
  total_observations: number
  total_events: number
  pending_events: number
  gpu_available: boolean
  device: string
  timestamp: string
}

export interface SystemStats {
  observations: { total: number; by_sensor: Record<string, number>; by_provenance: Record<string, number> }
  events: { total: number; pending: number; by_priority: Record<string, number>; by_status: Record<string, number> }
  models: {
    total: number; trained: number
    items: { id: string; name: string; sensor_type: string; status: string; evaluation_status: string; checkpoint_exists: boolean }[]
    radar_tracker: { type: string; available: boolean }
  }
  datasets: { total: number; items: { name: string; sensor_type: string; validation_status: string }[] }
  dependencies: Record<string, string | null>
  processing_mode: string
  replay: { status: string; processed_records: number; total_records: number; current_timestamp: string | null; replay_speed: number; queue_remaining: number; label: string }
  radar: Record<string, unknown>
  training_jobs: TrainingJobSummary[]
  evaluation_jobs: { id: string; status: string; model_id: string; dataset_name: string; map50: number | null; created_at: string }[]
  recent_observations: ObservationSummary[]
  recent_events: EventSummary[]
  latency: { component: string; latency_ms: number | null; timestamp: string }[]
  errors: LogEntry[]
  activity: LogEntry[]
}

// ─── Observations ────────────────────────────────────────────────────────────
export interface Observation {
  id: string
  sensor_type: 'satellite' | 'thermal' | 'radar'
  observation_timestamp: string | null
  is_utc: boolean
  source_dataset: string | null
  source_record_id: string | null
  object_class: string | null
  confidence: number | null
  bbox_x1: number | null
  bbox_y1: number | null
  bbox_x2: number | null
  bbox_y2: number | null
  latitude: number | null
  longitude: number | null
  coordinate_crs: string | null
  radar_track_id: string | null
  position_x: number | null
  position_y: number | null
  velocity_x: number | null
  velocity_y: number | null
  uncertainty: number | null
  model_name: string | null
  data_provenance: string
  processing_status: string
  image_path: string | null
  annotated_image_path: string | null
  image_url?: string | null
  model_version?: string | null
  checkpoint_id?: string | null
  metadata_json?: Record<string, unknown> | null
  created_at: string
}

export interface ObservationSummary {
  id: string
  sensor_type: string
  object_class: string | null
  confidence: number | null
  data_provenance: string
  observation_timestamp: string | null
  created_at: string
}

export interface ObservationListResponse {
  total: number
  limit: number
  offset: number
  observations: Observation[]
}

// ─── Events ──────────────────────────────────────────────────────────────────
export type ReviewPriority = 'LOW' | 'MEDIUM' | 'HIGH'
export type ReviewStatus = 'PENDING' | 'REVIEWING' | 'REVIEWED' | 'DISMISSED'

export interface Event {
  id: string
  review_priority: ReviewPriority
  source_sensors: string[] | null
  observation_ids: string[] | null
  evidence_summary: string | null
  uncertainty: number | null
  missing_information: string[] | null
  priority_explanation: string | null
  review_status: ReviewStatus
  reviewer_notes: string | null
  data_provenance: string
  created_at: string
  updated_at: string
  observations?: Observation[]
  correlations?: CorrelationDecision[]
  review_history?: { previous_status: string | null; new_status: string; notes: string | null; timestamp: string }[]
}

export interface EventSummary {
  id: string
  review_priority: ReviewPriority
  review_status: ReviewStatus
  source_sensors: string[] | null
  evidence_summary: string | null
  data_provenance?: string
  created_at: string
}

export interface EventListResponse {
  total: number
  events: Event[]
}

// ─── Correlation ─────────────────────────────────────────────────────────────
export type CorrelationStatus = 'CORRELATED' | 'NOT_CORRELATED' | 'INSUFFICIENT_METADATA'

export interface CorrelationDecision {
  id: string
  observation_ids: string[]
  decision_status: CorrelationStatus
  evidence: Record<string, unknown> | null
  reason: string | null
  missing_metadata: string[] | null
  temporal_delta_s: number | null
  spatial_distance_m: number | null
  config_params: Record<string, number> | null
  event_id: string | null
  created_at: string
}

export interface CorrelationResult {
  status: string
  message?: string
  correlations: CorrelationDecision[]
  duplicates: { observation_ids: string[]; sensor_type: string; temporal_delta_s: number | null; rule: string }[]
  groups?: { observation_ids: string[]; duplicate_ids: string[] }[]
  summary?: Record<string, number>
  total_pairs_checked: number
}

// ─── Datasets ────────────────────────────────────────────────────────────────
export type DatasetValidationStatus =
  | 'NOT_INSTALLED' | 'NOT_VALIDATED' | 'VALIDATED' | 'NOT_FOUND' | 'EMPTY'
  | 'INVALID' | 'RAW_NEEDS_CONVERSION' | 'CONVERTING' | 'CONVERSION_FAILED'

export interface Dataset {
  id: string
  name: string
  sensor_type: string
  source_url: string | null
  local_path: string | null
  annotation_format: string | null
  classes: string[] | null
  num_classes: number | null
  file_count: number | null
  train_count: number | null
  val_count: number | null
  test_count: number | null
  validation_status: DatasetValidationStatus
  validation_message: string | null
  access_requirements: string | null
  setup_instructions: string | null
  created_at: string
}

// ─── Models ──────────────────────────────────────────────────────────────────
export type ModelStatus = 'PRETRAINED' | 'TRAINING' | 'TRAINED' | 'FAILED'
export type EvalStatus = 'NOT_EVALUATED' | 'EVALUATING' | 'EVALUATED' | 'FAILED'

export interface ModelRecord {
  id: string
  name: string
  sensor_type: string
  architecture: string
  variant: string | null
  backbone: string | null
  neck: string | null
  head: string | null
  dataset_name: string | null
  pretrained: boolean
  trained: boolean
  checkpoint_path: string | null
  num_classes: number | null
  class_names: string[] | null
  input_size: number
  status: ModelStatus
  last_training_time: string | null
  evaluation_status: EvalStatus
  metrics: Record<string, number> | null
  known_limitations: string | null
  created_at: string
}

// ─── Training ────────────────────────────────────────────────────────────────
export type JobStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED'

export interface TrainingJob {
  id: string
  model_id: string
  dataset_name: string
  status: JobStatus
  epochs: number
  batch_size: number
  image_size: number
  learning_rate: number | null
  device: string
  current_epoch: number
  total_epochs: number
  train_loss: number | null
  val_loss: number | null
  best_map50: number | null
  checkpoint_path: string | null
  error_message: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string
  cancel_requested?: boolean
  epoch_history: Record<string, number>[]
  config?: Record<string, unknown>
  log_tail?: string
}

export interface TrainingJobSummary {
  id: string
  status: JobStatus
  model_id: string
  dataset_name?: string
  current_epoch: number
  total_epochs: number
  created_at: string
}

// ─── Evaluation ──────────────────────────────────────────────────────────────
export interface EvaluationJob {
  id: string
  model_id: string
  dataset_name: string
  dataset_split: string
  status: JobStatus
  num_images: number | null
  precision: number | null
  recall: number | null
  map50: number | null
  map50_95: number | null
  per_class_metrics: Record<string, Record<string, number | string>> | null
  confusion_matrix?: { labels: string[]; matrix: number[][]; axes: string } | null
  checkpoint_path?: string | null
  model_status_at_eval?: string | null
  warnings?: string[] | null
  avg_inference_ms: number | null
  error_message: string | null
  report_path: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string
}

// ─── Inference ───────────────────────────────────────────────────────────────
export interface Detection {
  class_id: number
  class_name: string
  confidence: number
  bbox_x1: number
  bbox_y1: number
  bbox_x2: number
  bbox_y2: number
}

export interface InferenceResult {
  status: string
  image_path: string
  annotated_image_path: string | null
  image_width: number
  image_height: number
  num_detections: number
  detections: Detection[]
  inference_time_ms: number
  model_name: string
  checkpoint: string
  confidence_threshold: number
  device: string
  timestamp: string
  warning?: string
  message?: string
  model_status?: string
  model_note?: string
  tiled?: boolean
  tile_size?: number | null
  class_names?: string[]
  warnings?: string[]
  observation_ids?: string[]
  image_url?: string
  annotated_image_url?: string
  modality_check?: { thermal_like: boolean; mean_saturation: number; method: string }
}

// ─── Radar ───────────────────────────────────────────────────────────────────
export interface RadarTrackHistory {
  timestamp: string
  source_id?: string | null
  truth_id?: string | null
  truth_x?: number | null
  truth_y?: number | null
  measured_x: number | null
  measured_y: number | null
  estimated_x: number
  estimated_y: number
  velocity_x: number
  velocity_y: number
  uncertainty: number
  measured: boolean
}

export interface RadarTrack {
  track_id: string
  status: 'ACTIVE' | 'TERMINATED'
  num_observations: number
  created_at: string
  last_update: string
  current_position: { x: number; y: number }
  current_velocity: { x: number; y: number }
  uncertainty: number
  duration_s: number
  history: RadarTrackHistory[]
}

export interface RadarProcessResult {
  status: string
  total_detections_processed: number
  batch_size: number
  active_tracks: number
  terminated_tracks: number
  new_tracks_created: number
  tracks_updated: number
  processing_time_ms: number
  timestamp: string
  tracks: RadarTrack[]
  coordinate_mode?: string
  units?: string
  observation_ids?: string[]
  warnings?: string[]
  report_id?: string
  evaluation?: Record<string, unknown> & { metrics_available: string[]; metrics_unavailable: string[] }
}

// ─── Replay ──────────────────────────────────────────────────────────────────
export type ReplayStatus = 'STOPPED' | 'LOADED' | 'PLAYING' | 'PAUSED' | 'COMPLETED'

export interface ReplayState {
  session_id: string | null
  status: ReplayStatus
  replay_speed: number
  total_records: number
  processed_records: number
  current_index: number
  current_timestamp: string | null
  start_timestamp: string | null
  end_timestamp: string | null
  data_provenance: string
  label: string
  progress?: number
  queue_preview?: { id: string | null; sensor_type: string | null; observation_timestamp: string }[]
  queue_remaining?: number
  skipped_records?: number
  stats?: { saved: number; rejected: number; events_created: number; events_updated: number }
  event_timeline?: { event_id: string; action: string; review_priority: string; recorded_timestamp: string; source_sensors: string[]; observation_count: number }[]
  errors?: string[]
  source_file?: string | null
  clock?: string
}

// ─── Synthetic ───────────────────────────────────────────────────────────────
export interface SyntheticScenarioResult {
  description: string
  label: string
  expected: Record<string, string>
  actual: {
    observations_saved: number
    validation: { id: string; status: string; errors: string[]; warnings: string[] }[]
    correlation_result: string | null
    has_duplicates: boolean | null
    event_priority: string | null
    event_id: string | null
    evidence_summary: string | null
    priority_explanation: string | null
  }
  checks: { check: string; expected: unknown; actual: unknown; pass: boolean }[]
  pass: boolean
}

export interface SyntheticRunResult {
  label: string
  total_scenarios: number
  passed: number
  failed: number
  run_tag?: string
  report_id?: string
  scenarios: Record<string, SyntheticScenarioResult>
}

// ─── Reports ─────────────────────────────────────────────────────────────────
export interface Report {
  id: string
  report_type: string
  timestamp: string
  test_mode: string
  source: string | null
  model_checkpoint: string | null
  num_samples: number
  metrics: Record<string, unknown> | null
  warnings: string[] | null
  limitations: string[] | null
  completion_status: string
  body?: Record<string, unknown>
}

// ─── Logs ────────────────────────────────────────────────────────────────────
export interface LogEntry {
  id?: string
  level: string
  component: string | null
  message: string
  timestamp: string | null
  details?: unknown
}
