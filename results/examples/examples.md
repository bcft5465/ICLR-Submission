# SAFER Benchmark — Representative Examples

Auto-extracted for paper Section 3 (Dataset Construction), addressing Reviewer CTbs and Reviewer 76GN's request for concrete image-claim examples.

## Clean example (id=openi_3146, source=openi)

![clean example](results/v1/examples/example_clean.png)

**Original (faithful) report text:**
> No focal lung consolidation. Heart size and pulmonary vascularity are within normal limits. No pneumothorax or pleural effusion. Osseous structures are grossly intact. No acute cardiopulmonary process.

**Claim shown to model:** identical to original (clean/faithful sample, no perturbation applied).

---

## Object example (id=mimiccxr_13633590_s59858604, source=mimiccxr)

![object example](results/v1/examples/example_object.jpg)

**Original (faithful) report text:**
> Findings: A left PICC tip projects over the low SVC.  There is no pleural effusion or pneumothorax.  There is no focal lung consolidation.  Cardiomediastinal silhouette is normal. There is no acute osseous abnormality. Impression: Left PICC ends in the low SVC.

**Injected claim shown to model:**
> Findings: A left PICC tip projects over the low SVC.  There is no pleural effusion or pneumothorax.  There is no focal liver consolidation.  Cardiomediastinal silhouette is normal. There is no acute osseous abnormality. Impression: Left PICC ends in the low SVC.

**Perturbation:** `lung` → `liver` (type: object)

---

## Attribute example (id=openi_3546, source=openi)

![attribute example](results/v1/examples/example_attribute.png)

**Original (faithful) report text:**
> Unchanged cardiomegaly. There is continued interstitial prominence bilaterally. Unchanged vascular appearance. There is patchy retrocardiac opacity. Negative for pneumothorax. Unchanged appearance of the chest with interstitial prominence the differential of which is XXXX but could include interstitial edema, infectious process or interstitial disease.

**Injected claim shown to model:**
> Unchanged cardiomegaly. There is continued interstitial prominence unilaterally. Unchanged vascular appearance. There is patchy retrocardiac opacity. Negative for pneumothorax. Unchanged appearance of the chest with interstitial prominence the differential of which is XXXX but could include interstitial edema, infectious process or interstitial disease.

**Perturbation:** `bilateral` → `unilateral` (type: attribute)

---

## Relational example (id=mimiccxr_16811873_s52639211, source=mimiccxr)

![relational example](results/v1/examples/example_relational.jpg)

**Original (faithful) report text:**
> Findings: The lungs are clear without consolidation or edema.  There is no pleural effusion or pneumothorax.  The cardiomediastinal silhouette is normal. Surgical changes are noted in the thyroid gland and right shoulder. Impression: No acute cardiopulmonary process.  Results were discussed with Dr. ___ at 12:05 p.m. on ___ via telephone by Dr. ___ at the time the findings were discovered.

**Injected claim shown to model:**
> Findings: The lungs are clear without consolidation or edema.  There is no pleural effusion or pneumothorax.  The cardiomediastinal silhouette is normal. Surgical changes are noted in the thyroid gland and left shoulder. Impression: No acute cardiopulmonary process.  Results were discussed with Dr. ___ at 12:05 p.m. on ___ via telephone by Dr. ___ at the time the findings were discovered.

**Perturbation:** `right` → `left` (type: relational)

---
