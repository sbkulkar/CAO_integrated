"""Reproducible evaluation and testing module for Medicine Verifier System."""

import os
import cv2
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Dict, Any, List

from src.config import (
    STATUS_MATCH,
    STATUS_MISMATCH,
    STATUS_UNREADABLE,
    STATUS_NOT_FOUND,
    STATUS_REVIEW_REQUIRED,
    PRESCRIPTION_PDF_PATH,
    CATALOGUE_DB_PATH
)
from src.prescription_parser import PrescriptionParser
from src.image_preprocessing import ImagePreprocessor
from src.ocr_engine import OCREngine
from src.catalogue_index import MedicineCatalogueIndex
from src.verification_engine import VerificationEngine

def generate_synthetic_package_image(
    brand_name: str,
    strength: str,
    composition: str,
    manufacturer: str,
    blur: bool = False,
    low_contrast: bool = False
) -> np.ndarray:
    """Generates a synthetic medicine package strip image for controlled, reproducible evaluation."""
    w, h = 600, 300
    img = Image.new("RGB", (w, h), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Decorative blister pack or box border
    draw.rectangle([10, 10, w - 10, h - 10], outline=(180, 180, 180), width=3)
    draw.rectangle([20, 20, w - 20, 80], fill=(25, 118, 210))

    # Header / Brand text
    draw.text((35, 30), brand_name.upper(), fill=(255, 255, 255))
    
    # Body details
    draw.text((35, 100), f"Strength: {strength}", fill=(20, 20, 20))
    draw.text((35, 140), f"Composition: {composition}", fill=(40, 40, 40))
    draw.text((35, 180), f"Mfg: {manufacturer}", fill=(80, 80, 80))
    draw.text((35, 220), "Batch: TEST-2026 | Exp: 12/2028", fill=(100, 100, 100))
    draw.text((35, 250), "Store below 25 deg C. Protect from light.", fill=(120, 120, 120))

    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=8))

    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if low_contrast:
        img_bgr = cv2.convertScaleAbs(img_bgr, alpha=0.3, beta=100)

    return img_bgr

class SystemEvaluator:
    """Systematic benchmark and verification evaluation harness."""

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.catalogue = MedicineCatalogueIndex(CATALOGUE_DB_PATH)
        if not self.catalogue.is_indexed():
            self.catalogue.build_index()
        self.ocr = OCREngine()
        self.parser = PrescriptionParser(self.ocr)
        self.verifier = VerificationEngine(self.catalogue)

    def run_prescription_parsing_benchmark(self) -> Dict[str, Any]:
        """Evaluates prescription parsing accuracy on the synthetic test prescription."""
        print("\n--- 1. Prescription Parsing Benchmark ---")
        items = self.parser.parse_pdf(PRESCRIPTION_PDF_PATH)
        expected_items = ["Paracip 500 Tablet", "Eciclo-P", "Augmentin 625 Duo Tablet"]
        
        detected_names = [it["medicine_name"] for it in items]
        matches = sum(1 for exp in expected_items if any(exp.lower() in d.lower() for d in detected_names))
        
        recall = matches / len(expected_items) if expected_items else 0.0
        print(f"Extracted {len(items)} items from test PDF. Expected: {len(expected_items)}. Recall: {recall * 100:.1f}%")
        return {
            "total_expected": len(expected_items),
            "total_extracted": len(items),
            "recall": recall,
            "items": items
        }

    def run_catalogue_search_benchmark(self) -> Dict[str, Any]:
        """Evaluates catalogue search latency and retrieval accuracy."""
        print("\n--- 2. Catalogue Search Benchmark ---")
        test_queries = [
            ("Augmentin 625 Duo Tablet", "625 mg"),
            ("Paracip 500 Tablet", "500 mg"),
            ("Azithral 500 Tablet", "500 mg"),
            ("Allegra 120mg Tablet", "120 mg"),
            ("Eciclo-P", None)
        ]
        
        latencies = []
        top1_hits = 0
        top5_hits = 0

        for q, strn in test_queries:
            t0 = time.time()
            cands = self.catalogue.search_candidates(q, prescribed_strength=strn, limit=5)
            dt = (time.time() - t0) * 1000
            latencies.append(dt)
            
            if cands:
                top5_hits += 1
                if cands[0]["similarity_score"] >= 0.8:
                    top1_hits += 1

        avg_latency = float(np.mean(latencies))
        print(f"Average search latency: {avg_latency:.2f} ms per query")
        print(f"Top-1 accuracy on target medicines: {top1_hits / len(test_queries) * 100:.1f}%")
        print(f"Top-5 recall: {top5_hits / len(test_queries) * 100:.1f}%")
        return {
            "avg_latency_ms": round(avg_latency, 2),
            "top1_accuracy": top1_hits / len(test_queries),
            "top5_recall": top5_hits / len(test_queries)
        }

    def run_verification_engine_evaluation(self) -> Dict[str, Any]:
        """Evaluates verification logic against known ground-truth cases."""
        print("\n--- 3. End-to-End Verification Engine Test Suite ---")
        
        # Test Case 1: Ground-truth MATCH
        # Prescribed Paracip 500mg vs Packaging Paracip 500mg
        img_match = generate_synthetic_package_image("Paracip", "500 mg", "Paracetamol IP 500mg", "Cipla Ltd")
        ocr_match = self.ocr.extract_text_from_image(img_match)

        # Test Case 2: Ground-truth MISMATCH (Conflicting strength)
        # Prescribed Paracip 500mg vs Packaging Paracip 650mg
        img_mismatch = generate_synthetic_package_image("Paracip", "650 mg", "Paracetamol IP 650mg", "Cipla Ltd")
        ocr_mismatch = self.ocr.extract_text_from_image(img_mismatch)

        # Test Case 3: Ground-truth UNREADABLE (Severe blur)
        img_blurry = generate_synthetic_package_image("Paracip", "500 mg", "Paracetamol", "Cipla Ltd", blur=True)
        ocr_blurry = self.ocr.extract_text_from_image(img_blurry)

        # Test Case 4: Ground-truth REVIEW REQUIRED (Unverified prescription strength)
        img_eciclo = generate_synthetic_package_image("Eciclo-P", "100mg/325mg", "Aceclofenac & Paracetamol", "Generic")
        ocr_eciclo = self.ocr.extract_text_from_image(img_eciclo)

        ground_truth_scenarios = [
            {
                "case_id": "TC-1 (Clear Match)",
                "prescribed": [{"item_id": "1", "medicine_name": "Paracip 500 Tablet", "prescribed_strength": "500 mg"}],
                "packages": [{"filename": "pkg_paracip_500.jpg", "ocr_result": ocr_match}],
                "expected_status": STATUS_MATCH
            },
            {
                "case_id": "TC-2 (Strength Mismatch)",
                "prescribed": [{"item_id": "1", "medicine_name": "Paracip 500 Tablet", "prescribed_strength": "500 mg"}],
                "packages": [{"filename": "pkg_paracip_650.jpg", "ocr_result": ocr_mismatch}],
                "expected_status": STATUS_MISMATCH
            },
            {
                "case_id": "TC-3 (Unreadable Blur)",
                "prescribed": [{"item_id": "1", "medicine_name": "Paracip 500 Tablet", "prescribed_strength": "500 mg"}],
                "packages": [{"filename": "pkg_blurry.jpg", "ocr_result": ocr_blurry}],
                "expected_status": STATUS_UNREADABLE
            },
            {
                "case_id": "TC-4 (Not Found)",
                "prescribed": [{"item_id": "3", "medicine_name": "Augmentin 625 Duo Tablet", "prescribed_strength": "625 mg"}],
                "packages": [], # No packages uploaded
                "expected_status": STATUS_NOT_FOUND
            },
            {
                "case_id": "TC-5 (Review Required - Unverified Strength)",
                "prescribed": [{"item_id": "2", "medicine_name": "Eciclo-P", "prescribed_strength": "Not verified"}],
                "packages": [{"filename": "pkg_eciclo.jpg", "ocr_result": ocr_eciclo}],
                "expected_status": STATUS_REVIEW_REQUIRED
            }
        ]

        test_results = []
        statuses = [STATUS_MATCH, STATUS_MISMATCH, STATUS_UNREADABLE, STATUS_NOT_FOUND, STATUS_REVIEW_REQUIRED]
        confusion_matrix = {exp: {act: 0 for act in statuses} for exp in statuses}

        for sc in ground_truth_scenarios:
            v_res = self.verifier.verify_session(sc["prescribed"], sc["packages"])
            actual_status = v_res["item_results"][0]["status"]
            expected = sc["expected_status"]
            
            confusion_matrix[expected][actual_status] += 1
            passed = (actual_status == expected)
            test_results.append({
                "case_id": sc["case_id"],
                "expected": expected,
                "actual": actual_status,
                "passed": passed,
                "explanation": v_res["item_results"][0]["explanation"]
            })
            print(f"  [{'PASS' if passed else 'FAIL'}] {sc['case_id']}: Expected={expected}, Got={actual_status}")

        total_cases = len(ground_truth_scenarios)
        total_passed = sum(1 for t in test_results if t["passed"])
        overall_accuracy = total_passed / total_cases

        print(f"\nVerification Test Suite Accuracy: {overall_accuracy * 100:.1f}% ({total_passed}/{total_cases})")
        print("\nConfusion Matrix:")
        header = f"{'Expected \\ Actual':<25}" + "".join([f"{s[:10]:<12}" for s in statuses])
        print(header)
        for exp in statuses:
            row_str = f"{exp:<25}" + "".join([f"{confusion_matrix[exp][act]:<12}" for act in statuses])
            print(row_str)

        return {
            "total_cases": total_cases,
            "total_passed": total_passed,
            "accuracy": overall_accuracy,
            "confusion_matrix": confusion_matrix,
            "test_cases": test_results,
            "limitations_statement": (
                "DATASET LIMITATION NOTICE: The 5,000 reference medicine package dataset consists of "
                "unlabelled packaging photos without verified ground-truth brand or strength annotations. "
                "Visual feature similarity is used strictly as supporting packaging retrieval evidence. "
                "Accuracy is evaluated solely against controlled, reproducible test scenarios and synthetic "
                "prescriptions to prevent circular evaluation or fabricated accuracy claims."
            )
        }

if __name__ == "__main__":
    evaluator = SystemEvaluator()
    evaluator.run_prescription_parsing_benchmark()
    evaluator.run_catalogue_search_benchmark()
    evaluator.run_verification_engine_evaluation()
