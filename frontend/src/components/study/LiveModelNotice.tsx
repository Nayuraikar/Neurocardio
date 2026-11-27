import { RR_MODELS, type Study } from '../../api/study'
import { PartialNotice } from '../states'
import { modelName, primaryOf } from './names'

/** Which model actually runs in the app. Shown wherever study scores sit near the app's own model. */
export function LiveModelNotice({ study }: { study: Study | null }) {
  const primary = study ? primaryOf(study) : null
  return <PartialNotice>
    <p style={{ margin: 0 }} data-testid="live-model-notice"><strong>The model running in this app is the deployed CNN-BiLSTM. It is not the multi-database study’s primary model{primary ? <> ({modelName(primary)})</> : null}.</strong>{' '}
      No study model, and none of the thresholds, temperatures or review cut-offs frozen by the study, is applied to uploaded recordings.{' '}
      {primary && <>The API cannot serve {modelName(primary)}: the API loads only the deployed Keras model, and this repository holds no model file for the study’s primary.
        {RR_MODELS.includes(primary) && ' It also needs RR-interval features that use the following beat and the whole-record median, which the study computed from reference beat positions.'}</>}</p>
  </PartialNotice>
}
