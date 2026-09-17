"""
End-to-End full pipeline integration test.
Verifies DATA -> CLIMATOLOGY -> EFI -> DETECTION -> TRACKING -> UNCERTAINTY -> VERIFICATION -> ALERT -> DB.
"""

import unittest
from backend.app.pipeline.orchestrator import orchestrator
from backend.app.db.database import SessionLocal
from backend.app.db import crud

class TestEndToEndPipeline(unittest.TestCase):
    def test_full_pipeline_execution(self):
        db = SessionLocal()
        try:
            run_id = "test_e2e_full_run"
            res = orchestrator.run_full_pipeline(
                run_id=run_id,
                scenario_type="monsoon_depression",
                seed=31415,
                displacement_bias_km=20.0,
                intensity_bias_pct=-4.0,
                db=db
            )
            
            self.assertEqual(res["run_id"], run_id)
            self.assertGreater(res["num_tracks"], 0)
            self.assertIsNotNone(res["primary_track"])
            
            # Check verification metrics were computed
            ver = res["verification"]
            self.assertGreater(ver["contingency"]["f1_score"], 0.3)
            self.assertGreater(ver["contingency"]["spatial_iou"], 0.2)
            self.assertGreater(ver["trajectory_error"]["mean_displacement_km"], 0.0)
            
            # Check alerts generated
            self.assertGreater(len(res["alerts"]), 0)
            
            # Check DB records
            run_rec = crud.get_run(db, run_id)
            self.assertIsNotNone(run_rec)
            self.assertEqual(run_rec.status, "COMPLETED")
            
            events = crud.get_events_for_run(db, run_id)
            self.assertGreater(len(events), 0)
            
            ver_rec = crud.get_verification_for_run(db, run_id)
            self.assertIsNotNone(ver_rec)
            self.assertAlmostEqual(ver_rec.f1_score, ver["contingency"]["f1_score"])
            
            prov_rec = crud.get_provenance_for_run(db, run_id)
            self.assertIsNotNone(prov_rec)
            self.assertEqual(prov_rec.random_seed, 31415)
        finally:
            db.close()

if __name__ == "__main__":
    unittest.main()
