import type { ModelMetrics } from '../api/metrics'
import { Num } from './Num'
import { Card, KeyValues, Table } from './ui'
import { PartialNotice } from './states'

export function SplitPanel({ m }: { m: ModelMetrics }) {
  const split = m.metrics.split, facts = m.split_facts
  return <Card title="Patient split of the app’s model">
    <Table head={['Part', 'Records', 'Patients', 'Record IDs']} rows={(['train', 'validation', 'test'] as const).map(part => [part,
      <Num key="r" v={split.records[part].length} src={`metrics#split.records.${part}.length`} />,
      <Num key="p" v={split.patient_group_ids[part].length} src={`metrics#split.patient_group_ids.${part}.length`} />,
      <span key="i" className="num small">{split.records[part].join(' ')}</span>])} />
    <KeyValues rows={[
      ['Test records disjoint from training and validation', facts.records_disjoint ? 'yes (checked now)' : 'NO: overlap found'],
      ['Test set equals de Chazal DS2 without record 202', facts.test_equals_ds2_without_202 === null ? 'could not be checked' : facts.test_equals_ds2_without_202 ? 'yes (checked now)' : 'no'],
      ['Excluded paced records', split.excluded_paced_records.join(', ')],
    ]} />
    {!facts.records_disjoint && <PartialNotice>Held-out results below are not valid while splits overlap.</PartialNotice>}
    <p>Project files record that model selection and calibration used validation data. <strong>Whether test data stayed unused during tuning cannot be verified</strong>, because the original training code is not in the repository.</p>
    <p><strong>How the test set was chosen is unverified.</strong> {facts.split_file_statement && <>A split file describes a class-count-constrained random draw. </>}
      {facts.test_equals_ds2_without_202 && <>The test set matches a published fixed partition, which such a draw would not be expected to produce. </>}</p>
    <p>Test scores exist for <Num v={m.variant_test_scores.length} src="metrics#variant_test_scores.length" /> model variants in the project files, not only for the selected one. DS2 is therefore not an untouched test set.</p>
  </Card>
}
