export const NAV: { group: string; items: [string, string][] }[] = [
  { group: 'Analyse', items: [['/studio', 'Analysis Studio'], ['/viewer', 'ECG Viewer'], ['/quality', 'Signal Quality'], ['/confidence', 'Confidence'], ['/xai', 'XAI Studio'], ['/errors', 'Misclassification Lab']] },
  { group: 'Study', items: [['/performance', 'Model Performance'], ['/comparison', 'Model Comparison'], ['/model-card', 'Model Card'], ['/dataset', 'Dataset Explorer']] },
  { group: 'Learn', items: [['/demo', 'Demo Playground'], ['/challenge', 'Challenge Mode']] },
  { group: 'Work', items: [['/history', 'History & Review'], ['/batch', 'Batch Analysis'], ['/compare', 'ECG Comparison'], ['/present', 'Presentation']] },
]
export const TITLES: Record<string, string> = Object.fromEntries(NAV.flatMap(g => g.items))
export const NEEDS_RECORDING = ['/studio', '/viewer', '/quality', '/confidence', '/xai', '/errors']
