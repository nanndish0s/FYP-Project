# Dataset Expansion Progress

## Current Status: IN PROGRESS ⏳

**Started**: December 23, 2025 at 22:16

### Phase 1: Mass Transcript Extraction (RUNNING)

**Script**: `src/data/mass_extract_transcripts.py`

**Target**: Extract transcripts from ALL CMU-MOSEI videos
- **Previous dataset**: 23 samples (from 200 attempted)
- **New attempt**: 1,275 unique YouTube video IDs
- **Expected completion**: ~13-15 minutes
- **Estimated success**: 100-150 transcripts (11-12% success rate based on previous)

**Features**:
- ✅ Processes ALL video IDs (not just sample)
- ✅ Progress saving every 100 videos
- ✅ Can resume if interrupted
- ✅ Detailed statistics tracking
- ✅ Error categorization

**Output Files**:
- `data/processed/all_transcripts.csv` - Successful extractions
- `data/processed/extraction_progress.json` - Progress tracker

---

## Next Steps (After Extraction Complete)

### Phase 2: LLM Labeling (4-6 hours)
**When**: After transcript extraction completes
**What**: Label all new transcripts with C3 scores using Groq API

**Steps**:
1. Run `src/labeling/generate_labels.py` on new transcripts
2. Target: Label 100-150 new samples
3. Cost: Free (Groq generous tier)
4. Time: ~4.8 seconds per transcript = ~8-12 minutes for 150

**Script to run**:
```bash
python src/labeling/generate_labels.py \
  --input data/processed/all_transcripts.csv \
  --output data/processed/all_with_c3_labels.csv
```

### Phase 3: Feature Extraction (2-3 hours)
**When**: After labeling complete
**What**: Extract 539 features for all new samples

**Steps**:
1. Acoustic features (from COVAREP - already extracted)
2. Lexical features (from transcripts)
3. Merge into ML-ready dataset

**Scripts to run**:
```bash
# Extract acoustic features
python src/features/extract_acoustic_features.py \
  --videos data/processed/all_transcripts.csv

# Extract lexical features
python src/features/extract_lexical_features.py \
  --input data/processed/all_with_c3_labels.csv

# Merge all features
python src/features/merge_features.py
```

### Phase 4: Model Retraining (1 hour)
**When**: After feature extraction complete
**What**: Retrain Random Forest models with larger dataset

**Expected improvement**:
- **Old**: 23 samples (18 train, 5 test)
- **New**: 100-150 samples (80-120 train, 20-30 test)
- **Impact**: Better generalization, more reliable metrics

**Script to run**:
```bash
python src/models/train_rf_models.py
```

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| 1. Transcript Extraction | 13-15 min | 🔄 **RUNNING** |
| 2. LLM Labeling | 8-12 min | ⏳ Pending |
| 3. Feature Extraction | 1-2 hours | ⏳ Pending |
| 4. Model Retraining | 30 min | ⏳ Pending |
| **Total** | **~3-4 hours** | |

---

## Expected Final Dataset

**Conservative Estimate** (11% success rate):
- New transcripts: ~140 samples
- Combined with existing: ~140 samples (replaced)
- **Final dataset**: ~140 samples

**Optimistic Estimate** (15% success rate):
- New transcripts: ~190 samples
- **Final dataset**: ~190 samples

**Impact**:
- **6-8x more training data** than current 23 samples
- Much more robust model performance
- Better cross-validation results
- Stronger project for submission

---

## Monitoring Progress

**Check current status**:
```bash
# View progress file
cat data/processed/extraction_progress.json

# Count transcripts so far
wc -l data/processed/all_transcripts.csv
```

**Resume if interrupted**:
- Script automatically loads progress and continues
- No need to restart from scratch

---

## Notes

- Script has been running since 22:16
- 0.5 second delay between requests (YouTube API rate limiting)
- Saves progress every 100 videos
- Can be stopped and resumed at any time

---

**Last Updated**: December 23, 2025 - 22:20
**Status**: Phase 1 in progress (7/1,275 videos processed)
