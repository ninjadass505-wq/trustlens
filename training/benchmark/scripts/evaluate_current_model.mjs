/**
 * TRUSTLENS V2 — Baseline Model Evaluator
 * Runs the current production detector (capcheck/ai-image-detection ViT-Base ONNX q4)
 * over the benchmark dataset manifest using exact TRUSTLENS multi-scale preprocessing.
 */

import { AutoModelForImageClassification, AutoProcessor, RawImage, env } from '@huggingface/transformers';
import fs from 'fs';
import path from 'path';

env.localModelPath = './frontend/models/';
env.allowRemoteModels = false;
env.allowLocalModels = true;

const MODEL_PATH = './frontend/models/onnx-community/ai-image-detection-ONNX';

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function aggregatePatches(patchResults) {
  if (!patchResults || patchResults.length === 0) {
    return {
      syntheticScore: 50,
      realScore: 50,
      scoreGap: 0,
      patchStdDev: 0,
      disagreementRange: 0,
      evidenceConsistency: 'LOW',
      finalDecision: 'INCONCLUSIVE'
    };
  }

  const synthScores = patchResults.map((p) => p.syntheticScore);
  const n = synthScores.length;

  const minSynth = Math.min(...synthScores);
  const maxSynth = Math.max(...synthScores);
  const disagreementRange = maxSynth - minSynth;

  const meanSynth = synthScores.reduce((a, b) => a + b, 0) / n;
  const variance = synthScores.reduce((acc, val) => acc + Math.pow(val - meanSynth, 2), 0) / n;
  const patchStdDev = Math.sqrt(variance);

  const sortedSynth = [...synthScores].sort((a, b) => a - b);
  const medianSynth = n % 2 === 0
    ? (sortedSynth[n / 2 - 1] + sortedSynth[n / 2]) / 2
    : sortedSynth[Math.floor(n / 2)];

  let trimmedSynth = meanSynth;
  if (n >= 4) {
    const trimmedList = sortedSynth.slice(1, sortedSynth.length - 1);
    trimmedSynth = trimmedList.reduce((a, b) => a + b, 0) / trimmedList.length;
  }

  const aggregateSynth = Math.round(0.7 * trimmedSynth + 0.3 * medianSynth);
  const aggregateReal = 100 - aggregateSynth;
  const scoreGap = Math.abs(aggregateSynth - aggregateReal);

  let evidenceConsistency = 'HIGH';
  if (patchStdDev > 20 || disagreementRange > 38) {
    evidenceConsistency = 'LOW';
  } else if (patchStdDev > 12 || disagreementRange > 22) {
    evidenceConsistency = 'MEDIUM';
  }

  let finalDecision = 'INCONCLUSIVE';
  if (evidenceConsistency === 'LOW') {
    finalDecision = 'INCONCLUSIVE';
  } else if (scoreGap < 18) {
    finalDecision = 'INCONCLUSIVE';
  } else if (aggregateReal > aggregateSynth) {
    finalDecision = aggregateReal >= 65 ? 'LIKELY REAL' : 'NEEDS VERIFICATION';
  } else {
    finalDecision = aggregateSynth >= 65 ? 'LIKELY AI-GENERATED' : 'NEEDS VERIFICATION';
  }

  return {
    syntheticScore: aggregateSynth,
    realScore: aggregateReal,
    scoreGap,
    meanSynth: Math.round(meanSynth * 10) / 10,
    medianSynth: Math.round(medianSynth * 10) / 10,
    minSynth,
    maxSynth,
    patchStdDev: Math.round(patchStdDev * 10) / 10,
    disagreementRange,
    evidenceConsistency,
    finalDecision
  };
}

async function extractViews(imagePath) {
  const img = await RawImage.read(imagePath);
  const origW = img.width;
  const origH = img.height;
  const views = [];

  // 1. Aspect-ratio-preserving Center Crop Global Context (Zero gray bars)
  const targetSize = 224;
  const globalScale = Math.max(targetSize / origW, targetSize / origH);
  const scaledW = Math.max(targetSize, Math.round(origW * globalScale));
  const scaledH = Math.max(targetSize, Math.round(origH * globalScale));
  const resized = await img.resize(scaledW, scaledH);
  const globalCrop = await resized.center_crop(targetSize, targetSize);
  views.push({ id: 'global_center', name: 'Global Context (Center)', image: globalCrop, type: 'global' });

  // 2. Native 1:1 Scale Patches (unscaled)
  if (origW >= targetSize && origH >= targetSize) {
    const cx = Math.round((origW - targetSize) / 2);
    const cy = Math.round((origH - targetSize) / 2);
    const centerPatch = await img.crop([cx, cy, targetSize, targetSize]);
    views.push({ id: 'native_center', name: 'Center Focal Patch (1:1)', image: centerPatch, type: 'native' });

    const ulX = Math.round(origW * 0.12);
    const ulY = Math.round(origH * 0.12);
    const ulPatch = await img.crop([ulX, ulY, targetSize, targetSize]);
    views.push({ id: 'native_upper_left', name: 'Upper-Left Patch (1:1)', image: ulPatch, type: 'native' });

    const brX = Math.max(0, Math.round(origW * 0.88 - targetSize));
    const brY = Math.max(0, Math.round(origH * 0.88 - targetSize));
    const brPatch = await img.crop([brX, brY, targetSize, targetSize]);
    views.push({ id: 'native_lower_right', name: 'Lower-Right Patch (1:1)', image: brPatch, type: 'native' });

    const txX = Math.max(0, Math.min(origW - targetSize, Math.round(origW * 0.5 - targetSize / 2)));
    const txY = Math.max(0, Math.min(origH - targetSize, Math.round(origH * 0.3)));
    const txPatch = await img.crop([txX, txY, targetSize, targetSize]);
    views.push({ id: 'native_texture', name: 'Texture Detail Patch (1:1)', image: txPatch, type: 'native' });
  } else {
    const scaledUp = await img.resize(targetSize, targetSize);
    views.push({ id: 'native_scaled', name: 'Scaled View', image: scaledUp, type: 'native' });
  }

  return { origW, origH, views };
}

async function main() {
  const args = process.argv.slice(2);
  const manifestPath = args[0] || 'training/benchmark/dataset/manifest.json';
  const datasetRoot = args[1] || 'training/benchmark/dataset/';
  const outputPath = args[2] || 'training/benchmark/results/evaluation_raw.json';

  console.log('====================================================');
  console.log('TRUSTLENS BENCHMARK — CURRENT MODEL BASELINE EVALUATION');
  console.log('====================================================');
  console.log(`Manifest: ${manifestPath}`);
  console.log(`Model:    ${MODEL_PATH} (ONNX q4)`);

  if (!fs.existsSync(manifestPath)) {
    console.error(`Error: Manifest file not found at ${manifestPath}`);
    process.exit(1);
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  console.log(`Loaded ${manifest.length} items from manifest.`);

  console.log('\nInitializing ONNX Runtime and Preprocessor...');
  const model = await AutoModelForImageClassification.from_pretrained(MODEL_PATH, {
    device: 'cpu',
    dtype: 'q4'
  });
  const processor = await AutoProcessor.from_pretrained(MODEL_PATH);
  console.log('Detector initialized successfully.\n');

  const results = [];
  let processed = 0;
  let missing = 0;

  for (const item of manifest) {
    const fullPath = path.join(datasetRoot, item.path);
    if (!fs.existsSync(fullPath)) {
      missing++;
      console.warn(`[SKIP] Image not found on disk: ${fullPath} (Item ID: ${item.id})`);
      continue;
    }

    try {
      const { origW, origH, views } = await extractViews(fullPath);
      const patchResults = [];

      for (const view of views) {
        const inputs = await processor(view.image);
        const outputs = await model(inputs);
        const logits = Array.from(outputs.logits.data);
        const probs = softmax(logits);

        // Class 0 = REAL, Class 1 = FAKE / SYNTHETIC
        const realScore = Math.round(probs[0] * 100);
        const syntheticScore = Math.round(probs[1] * 100);

        patchResults.push({
          id: view.id,
          name: view.name,
          type: view.type,
          logits,
          realScore,
          syntheticScore
        });
      }

      const agg = aggregatePatches(patchResults);

      results.push({
        id: item.id,
        path: item.path,
        ground_truth: item.label,
        source: item.source,
        generator: item.generator || null,
        generator_family: item.generator_family || null,
        camera_device: item.camera_device || null,
        resolution: item.resolution || `${origW}x${origH}`,
        aspect_ratio: item.aspect_ratio || `${(origW / origH).toFixed(2)}:1`,
        compression: item.compression || 'unknown',
        scene_category: item.scene_category || 'general',
        split: item.split,
        group_id: item.group_id || null,
        inference: {
          real_score: agg.realScore,
          synthetic_score: agg.syntheticScore,
          score_gap: agg.scoreGap,
          mean_synth: agg.meanSynth,
          median_synth: agg.medianSynth,
          min_synth: agg.minSynth,
          max_synth: agg.maxSynth,
          patch_std_dev: agg.patchStdDev,
          disagreement_range: agg.disagreementRange,
          evidence_consistency: agg.evidenceConsistency,
          final_decision: agg.finalDecision,
          patch_scores: patchResults
        }
      });

      processed++;
      if (processed % 5 === 0 || processed === manifest.length) {
        console.log(`Evaluated ${processed}/${manifest.length} images...`);
      }
    } catch (err) {
      console.error(`Error processing image ${item.id}:`, err);
    }
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2), 'utf8');

  console.log('\n====================================================');
  console.log(`Evaluation complete.`);
  console.log(`Processed: ${processed} images`);
  console.log(`Missing on disk: ${missing} images`);
  console.log(`Raw results saved to: ${outputPath}`);
  console.log('====================================================');
}

main().catch(console.error);
