export type DemoMeta = { id: string; database: string; record: string; sample: number; app_model_split: string; study_split: string; label?: string }
export type DemoSample = DemoMeta & { model_input: number[]; sampling_rate: number; model_version?: string; prediction: null | { predicted_class: string; probabilities: number[] } }
export type DemoList = { source: string; selection: string; eligibility: string; samples: DemoMeta[] }
