"""R3 admission/completeness tests without running the benchmark."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import analytic_coordinate_contract as a
import analytic_coordinate_r3 as r

class R3Tests(unittest.TestCase):
    def rows(self):
        return [{**case,'bridge':bridge,'arms':{arm:{'spill':0,'missing':0,'foreground_delta':0,'predicted_foreground':0,'reference_foreground':0,'iou':1.,'boundary_crossings':0,'centroid_dxdy':None} for arm in a.ARMS}} for case in a.cases() for bridge in a.BRIDGES]

    def test_complete_matrix_and_reject_missing_duplicate_category(self):
        rows=self.rows();r.validate_rows(rows)
        with self.assertRaisesRegex(ValueError,'CASE'):r.validate_rows(rows[:-1])
        rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'CASE'):r.validate_rows(rows)

    def test_invalid_numbers_and_conservation_fail_closed(self):
        rows=self.rows();rows[0]['arms']['legacy']['iou']=float('nan')
        with self.assertRaisesRegex(ValueError,'NUMERIC'):r.validate_rows(rows)
        rows=self.rows();rows[0]['arms']['legacy']['spill']=1
        with self.assertRaisesRegex(ValueError,'CONSERVATION'):r.validate_rows(rows)

    def test_trace_shape_rejection(self):
        case=next(a.cases())
        values={k:__import__('numpy').zeros(shape) for k,shape in r.trace_shapes(case).items()}
        r.validate_trace_arrays(values,case)
        values['pooled_aligned']=__import__('numpy').zeros((3,27,28))
        with self.assertRaisesRegex(ValueError,'TRACE_SHAPE'):r.validate_trace_arrays(values,case)

    def test_partial_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'measurements.jsonl').write_text(json.dumps(self.rows()[0])+'\n')
            with self.assertRaises(ValueError):r.verify_complete(out,{})

    def test_hash_drift_and_independent_review_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input';p.write_text('frozen')
            policy={'sources_sha256':{str(p):r.digest(p)}}
            r.verify_bindings(policy)
            p.write_text('drift')
            with self.assertRaisesRegex(ValueError,'SOURCE_DRIFT'):r.verify_bindings(policy)
        policy_sha='a'*64
        review={'decision':'APPROVE_R3_SYNTHETIC_ONLY','reviewer':'/root/reviewer','preregistration_sha256':policy_sha}
        r.verify_review(review,policy_sha)
        for key,value in [('reviewer','/root'),('preregistration_sha256','b'*64),('decision','PENDING')]:
            bad={**review,key:value}
            with self.assertRaisesRegex(ValueError,'REVIEW'):r.verify_review(bad,policy_sha)

    def test_exclusive_output_preserves_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'keep').write_text('unchanged')
            with self.assertRaisesRegex(ValueError,'OUTPUT'):r.exclusive_output(out)
            self.assertEqual((out/'keep').read_text(),'unchanged')

if __name__=='__main__':unittest.main()
