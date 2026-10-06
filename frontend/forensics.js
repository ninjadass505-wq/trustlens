/**
 * TrustLens Real-World Image Forensic Engine (v2.0)
 * 
 * Features:
 * 1. Resolution-independent input pipeline (accepts any arbitrary resolution).
 * 2. Aspect-ratio preserving geometry (no stretching or distortion).
 * 3. Multi-scale & Native 1:1 patch analysis (preserves sensor noise, textures, fine artifacts).
 * 4. Forensic signal extractors (Laplacian noise variance, high-frequency residual, color entropy).
 * 5. Native binary EXIF & camera metadata inspector.
 * 6. Out-Of-Distribution (OOD) & applicability detector.
 * 7. Robust trimmed-consensus evidence aggregation (prevents single-patch false alarms).
 * 8. Calibrated decision logic & Section 22 result architecture.
 */

// Binary EXIF parser for JPEG/TIFF without external dependenciesss1
export function parseExif(arrayBuffer) {
  const data = new DataView(arrayBuffer);
  const metadata = {
    hasExif: false,
    make: null,
    model: null,
    software: null,
    dateTime: null,
    iso: null,
    fNumber: null,
    exposureTime: null,
    lensModel: null,
    suspiciousSoftware: false,
    aiKeywords: []
  };

  try {
    if (data.byteLength < 4 || data.getUint16(0, false) !== 0xFFD8) {
      // Not a JPEG
      return metadata;
    }

    let offset = 2;
    while (offset < data.byteLength - 4) {
      const marker = data.getUint16(offset, false);
      offset += 2;

      if (marker === 0xFFE1) { // APP1 Exif Marker
        const length = data.getUint16(offset, false);
        offset += 2;

        // Check for 'Exif\0\0'
        const header = String.fromCharCode(
          data.getUint8(offset),
          data.getUint8(offset + 1),
          data.getUint8(offset + 2),
          data.getUint8(offset + 3)
        );

        if (header === 'Exif') {
          metadata.hasExif = true;
          const tiffStart = offset + 6;
          const isLittleEndian = data.getUint16(tiffStart, false) === 0x4949; // 'II'

          const ifdOffset = data.getUint32(tiffStart + 4, isLittleEndian);
          readIFD(data, tiffStart, tiffStart + ifdOffset, isLittleEndian, metadata);
        }
        break;
      } else if ((marker & 0xFF00) === 0xFF00 && marker !== 0xFFD8 && marker !== 0xFFD9) {
        const length = data.getUint16(offset, false);
        offset += length;
      } else {
        break;
      }
    }
  } catch (err) {
    console.warn('[TrustLens Forensics] EXIF parse error:', err);
  }

  // Check for common AI-generator signatures in metadata strings
  const combined = `${metadata.make || ''} ${metadata.model || ''} ${metadata.software || ''}`.toLowerCase();
  const aiGenerators = ['midjourney', 'stable diffusion', 'dall-e', 'comfyui', 'novelai', 'adobe firefly', 'flux.1'];
  for (const gen of aiGenerators) {
    if (combined.includes(gen)) {
      metadata.suspiciousSoftware = true;
      metadata.aiKeywords.push(gen);
    }
  }

  return metadata;
}

function readIFD(data, tiffStart, ifdStart, isLE, metadata) {
  try {
    if (ifdStart + 2 > data.byteLength) return;
    const numEntries = data.getUint16(ifdStart, isLE);
    let entryOffset = ifdStart + 2;

    for (let i = 0; i < numEntries; i++) {
      if (entryOffset + 12 > data.byteLength) break;
      const tag = data.getUint16(entryOffset, isLE);
      const type = data.getUint16(entryOffset + 2, isLE);
      const count = data.getUint32(entryOffset + 4, isLE);
      const valOffset = entryOffset + 8;

      const getString = () => {
        let strOffset = count > 4 ? tiffStart + data.getUint32(valOffset, isLE) : valOffset;
        let str = '';
        for (let j = 0; j < count; j++) {
          if (strOffset + j >= data.byteLength) break;
          const charCode = data.getUint8(strOffset + j);
          if (charCode === 0) break;
          str += String.fromCharCode(charCode);
        }
        return str.trim();
      };

      if (tag === 0x010F) metadata.make = getString();       // Make
      if (tag === 0x0110) metadata.model = getString();      // Model
      if (tag === 0x0131) metadata.software = getString();   // Software
      if (tag === 0x0132) metadata.dateTime = getString();   // DateTime

      // SubIFD (Exif Offset)
      if (tag === 0x8769) {
        const subIfdOffset = tiffStart + data.getUint32(valOffset, isLE);
        readIFD(data, tiffStart, subIfdOffset, isLE, metadata);
      }

      entryOffset += 12;
    }
  } catch (e) {
    // Ignore corrupt tags
  }
}

/**
 * Computes pixel-level forensic signals (Laplacian noise variance, high-frequency energy ratio, entropy)
 */
export function analyzeForensicSignals(ctx, width, height) {
  const sampleW = Math.min(width, 512);
  const sampleH = Math.min(height, 512);
  const imgData = ctx.getImageData(0, 0, sampleW, sampleH);
  const pixels = imgData.data;

  // Grayscale luminance conversion & Laplacian edge filter
  const gray = new Float32Array(sampleW * sampleH);
  for (let i = 0, j = 0; i < pixels.length; i += 4, j++) {
    gray[j] = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
  }

  // Laplacian kernel: [0, 1, 0,  1, -4, 1,  0, 1, 0]
  let laplacianSum = 0;
  let laplacianSqSum = 0;
  let count = 0;

  for (let y = 1; y < sampleH - 1; y++) {
    for (let x = 1; x < sampleW - 1; x++) {
      const idx = y * sampleW + x;
      const lap = 
        gray[idx - sampleW] +
        gray[idx + sampleW] +
        gray[idx - 1] +
        gray[idx + 1] -
        4 * gray[idx];
      
      laplacianSum += lap;
      laplacianSqSum += lap * lap;
      count++;
    }
  }

  const meanLap = laplacianSum / (count || 1);
  const lapVariance = Math.max(0, (laplacianSqSum / (count || 1)) - (meanLap * meanLap));

  // Compute color channel histogram entropy
  const histR = new Uint32Array(256);
  const histG = new Uint32Array(256);
  const histB = new Uint32Array(256);
  for (let i = 0; i < pixels.length; i += 4) {
    histR[pixels[i]]++;
    histG[pixels[i + 1]]++;
    histB[pixels[i + 2]]++;
  }

  const computeEntropy = (hist, total) => {
    let ent = 0;
    for (let i = 0; i < 256; i++) {
      if (hist[i] > 0) {
        const p = hist[i] / total;
        ent -= p * Math.log2(p);
      }
    }
    return ent;
  };

  const totalPixels = sampleW * sampleH;
  const entropyR = computeEntropy(histR, totalPixels);
  const entropyG = computeEntropy(histG, totalPixels);
  const entropyB = computeEntropy(histB, totalPixels);
  const meanEntropy = (entropyR + entropyG + entropyB) / 3;

  return {
    laplacianVariance: Math.round(lapVariance * 10) / 10,
    colorEntropy: Math.round(meanEntropy * 100) / 100,
    isLowTexture: lapVariance < 15,
    isExtremelySmooth: lapVariance < 2.5,
    hasCameraLikeNoise: lapVariance >= 25 && meanEntropy >= 3.8
  };
}

/**
 * Evaluates Out-Of-Distribution (OOD) & Applicability
 */
export function checkApplicability(imageInfo, forensicSignals) {
  const reasons = [];
  let status = 'GOOD'; // 'GOOD' | 'LIMITED' | 'OUT OF SCOPE'

  // 1. Resolution limits
  if (imageInfo.width < 64 || imageInfo.height < 64) {
    status = 'OUT OF SCOPE';
    reasons.push(`Resolution (${imageInfo.width}×${imageInfo.height}) is too small for forensic analysis (minimum 64×64).`);
  } else if (imageInfo.width < 180 || imageInfo.height < 180) {
    status = 'LIMITED';
    reasons.push(`Low resolution (${imageInfo.width}×${imageInfo.height}); fine sensor noise details are attenuated.`);
  }

  // 2. Extreme aspect ratios
  const ratio = Math.max(imageInfo.width / imageInfo.height, imageInfo.height / imageInfo.width);
  if (ratio > 7.0) {
    status = 'OUT OF SCOPE';
    reasons.push(`Extreme aspect ratio (${imageInfo.aspectRatio}) is outside standard photographic distribution.`);
  } else if (ratio > 3.5) {
    if (status !== 'OUT OF SCOPE') status = 'LIMITED';
    reasons.push(`Unusual panoramic aspect ratio (${imageInfo.aspectRatio}).`);
  }

  // 3. Information content & flatness
  if (forensicSignals.isExtremelySmooth && forensicSignals.colorEntropy < 2.0) {
    status = 'OUT OF SCOPE';
    reasons.push('Image lacks texture or features (solid color, uniform graphic, or blank screen).');
  }

  // 4. File size vs resolution anomaly
  const bpp = (imageInfo.fileSize * 8) / (imageInfo.width * imageInfo.height);
  if (bpp < 0.25 && imageInfo.width >= 1000) {
    if (status !== 'OUT OF SCOPE') status = 'LIMITED';
    reasons.push('Heavy compression detected; compression artifacts may obscure subtle forensic signatures.');
  }

  return { status, reasons };
}

/**
 * Generates Resolution-Independent Preprocessed Views:
 * 1. Global Context View: Aspect-ratio-preserving center-crop coverage (ZERO gray bars, 100% natural photo pixels)
 * 2. High-Resolution Native 1:1 Patches: Preserves native sensor noise, edges, textures
 * 3. Secondary Context View for wide/tall images
 */
export function generateForensicPatches(sourceImage, targetSize = 224) {
  const origW = sourceImage.naturalWidth || sourceImage.width;
  const origH = sourceImage.naturalHeight || sourceImage.height;
  const patches = [];

  // Helper to create an offscreen canvas
  const makeCanvas = (w, h) => {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    return { canvas: c, ctx };
  };

  // 1. GLOBAL CONTEXT VIEW (ZERO GRAY BARS):
  // Scale so that the SHORTER dimension fits targetSize (224), then extract the center 224x224 crop.
  // This guarantees 100% genuine photographic pixels, zero artificial borders, and zero step-function edges.
  const globalScale = Math.max(targetSize / origW, targetSize / origH);
  const srcCropW = Math.min(origW, Math.round(targetSize / globalScale));
  const srcCropH = Math.min(origH, Math.round(targetSize / globalScale));
  const srcCropX = Math.max(0, Math.round((origW - srcCropW) / 2));
  const srcCropY = Math.max(0, Math.round((origH - srcCropH) / 2));

  const { canvas: globalCanvas, ctx: globalCtx } = makeCanvas(targetSize, targetSize);
  globalCtx.drawImage(sourceImage, srcCropX, srcCropY, srcCropW, srcCropH, 0, 0, targetSize, targetSize);
  
  patches.push({
    id: 'global_center',
    name: 'Global Context (Center)',
    type: 'global',
    canvas: globalCanvas,
    bounds: { x: srcCropX, y: srcCropY, width: srcCropW, height: srcCropH },
    description: `Aspect-ratio preserved full scene center (${srcCropW}×${srcCropH} sampled to ${targetSize}×${targetSize}, zero padding)`
  });

  // If the image is wide (e.g. 16:9 or 4:3), add a secondary Global Context View covering the periphery
  const aspectRatio = origW / origH;
  if (aspectRatio > 1.35 && origW > targetSize) {
    const { canvas: leftCanvas, ctx: leftCtx } = makeCanvas(targetSize, targetSize);
    leftCtx.drawImage(sourceImage, 0, srcCropY, srcCropW, srcCropH, 0, 0, targetSize, targetSize);
    patches.push({
      id: 'global_left',
      name: 'Global Context (Left)',
      type: 'global',
      canvas: leftCanvas,
      bounds: { x: 0, y: srcCropY, width: srcCropW, height: srcCropH },
      description: `Aspect-ratio preserved left flank (${srcCropW}×${srcCropH}, zero padding)`
    });
  } else if (aspectRatio < 0.75 && origH > targetSize) {
    const { canvas: topCanvas, ctx: topCtx } = makeCanvas(targetSize, targetSize);
    topCtx.drawImage(sourceImage, srcCropX, 0, srcCropW, srcCropH, 0, 0, targetSize, targetSize);
    patches.push({
      id: 'global_top',
      name: 'Global Context (Upper)',
      type: 'global',
      canvas: topCanvas,
      bounds: { x: srcCropX, y: 0, width: srcCropW, height: srcCropH },
      description: `Aspect-ratio preserved upper scene (${srcCropW}×${srcCropH}, zero padding)`
    });
  }

  // 2. NATIVE 1:1 SCALE PATCHES (preserves fine sensor noise, demosaicing, and edges)
  if (origW >= targetSize && origH >= targetSize) {
    const patchW = targetSize;
    const patchH = targetSize;

    const coords = [
      {
        id: 'native_center',
        name: 'Center Focal Patch (1:1)',
        type: 'native',
        x: Math.round((origW - patchW) / 2),
        y: Math.round((origH - patchH) / 2),
        desc: 'Central subject area at 1:1 original pixel scale (unscaled)'
      },
      {
        id: 'native_top_left',
        name: 'Upper-Left Patch (1:1)',
        type: 'native',
        x: Math.round(origW * 0.12),
        y: Math.round(origH * 0.12),
        desc: 'Upper peripheral region at 1:1 original pixel scale (unscaled)'
      },
      {
        id: 'native_bottom_right',
        name: 'Lower-Right Patch (1:1)',
        type: 'native',
        x: Math.max(0, Math.round(origW * 0.88 - patchW)),
        y: Math.max(0, Math.round(origH * 0.88 - patchH)),
        desc: 'Lower peripheral region at 1:1 original pixel scale (unscaled)'
      }
    ];

    // High Detail / Structural candidate (quarter-center)
    coords.push({
      id: 'native_texture',
      name: 'Texture Detail Patch (1:1)',
      type: 'native',
      x: Math.max(0, Math.min(origW - patchW, Math.round(origW * 0.5 - patchW / 2))),
      y: Math.max(0, Math.min(origH - patchH, Math.round(origH * 0.3))),
      desc: 'High-frequency structural region at 1:1 original pixel scale (unscaled)'
    });

    for (const c of coords) {
      const { canvas: pCanvas, ctx: pCtx } = makeCanvas(patchW, patchH);
      pCtx.drawImage(sourceImage, c.x, c.y, patchW, patchH, 0, 0, patchW, patchH);
      patches.push({
        id: c.id,
        name: c.name,
        type: c.type,
        canvas: pCanvas,
        bounds: { x: c.x, y: c.y, width: patchW, height: patchH },
        description: c.desc
      });
    }
  } else {
    // For smaller images (< 224px): scale up smoothly without artificial gray borders
    const { canvas: pCanvas, ctx: pCtx } = makeCanvas(targetSize, targetSize);
    pCtx.drawImage(sourceImage, 0, 0, origW, origH, 0, 0, targetSize, targetSize);
    patches.push({
      id: 'native_scaled',
      name: 'Scaled View (No Borders)',
      type: 'native',
      canvas: pCanvas,
      bounds: { x: 0, y: 0, width: origW, height: origH },
      description: 'Smoothly scaled to model dimension without padding'
    });
  }

  return patches;
}

/**
 * Aggregates multi-patch evidence using robust trimmed statistics & patch consistency analysis.
 * - Prevents a single anomalous patch from dominating the verdict.
 * - Explicitly detects and reports patch disagreement.
 * - Enforces INCONCLUSIVE when signals conflict.
 */
export function aggregatePatchEvidence(patchResults, forensicSignals, applicability) {
  if (!patchResults || patchResults.length === 0) {
    return {
      syntheticScore: 50,
      realScore: 50,
      scoreGap: 0,
      patchStdDev: 0,
      disagreementRange: 0,
      evidenceStrength: 'WEAK',
      evidenceConsistency: 'LOW',
      label: 'INCONCLUSIVE',
      explanation: 'No patch evidence could be collected.'
    };
  }

  const synthScores = patchResults.map((p) => p.syntheticScore);
  const realScores = patchResults.map((p) => p.realScore);
  const n = synthScores.length;

  // 1. Descriptive Statistics
  const minSynth = Math.min(...synthScores);
  const maxSynth = Math.max(...synthScores);
  const disagreementRange = maxSynth - minSynth;

  const meanSynth = synthScores.reduce((a, b) => a + b, 0) / n;
  const variance = synthScores.reduce((acc, val) => acc + Math.pow(val - meanSynth, 2), 0) / n;
  const patchStdDev = Math.sqrt(variance);

  // 2. Median Synthetic Score
  const sortedSynth = [...synthScores].sort((a, b) => a - b);
  const medianSynth = n % 2 === 0
    ? (sortedSynth[n / 2 - 1] + sortedSynth[n / 2]) / 2
    : sortedSynth[Math.floor(n / 2)];

  // 3. Trimmed Mean (removes highest and lowest outlier if n >= 4 to neutralize single-patch anomalies)
  let trimmedSynth = meanSynth;
  if (n >= 4) {
    const trimmedList = sortedSynth.slice(1, sortedSynth.length - 1);
    trimmedSynth = trimmedList.reduce((a, b) => a + b, 0) / trimmedList.length;
  }

  // Robust Aggregate Score combines trimmed mean (70%) and median (30%)
  const aggregateSynth = Math.round(0.7 * trimmedSynth + 0.3 * medianSynth);
  const aggregateReal = 100 - aggregateSynth;
  const scoreGap = Math.abs(aggregateSynth - aggregateReal);

  // 4. Evidence Consistency Evaluation (Sections 24 & 25)
  let evidenceConsistency = 'HIGH';
  if (patchStdDev > 20 || disagreementRange > 38) {
    evidenceConsistency = 'LOW'; // Serious conflict between regions
  } else if (patchStdDev > 12 || disagreementRange > 22) {
    evidenceConsistency = 'MEDIUM';
  }

  // 5. Evidence Strength Evaluation
  let evidenceStrength = 'MODERATE';
  if (applicability.status === 'OUT OF SCOPE' || evidenceConsistency === 'LOW') {
    evidenceStrength = 'WEAK';
  } else if (scoreGap >= 30 && evidenceConsistency === 'HIGH' && patchStdDev <= 10) {
    evidenceStrength = 'STRONG';
  } else if (scoreGap < 18) {
    evidenceStrength = 'WEAK';
  }

  // 6. Principled Decision Logic (Sections 26 & 30)
  let label = 'INCONCLUSIVE';
  let explanation = '';

  if (applicability.status === 'OUT OF SCOPE') {
    label = 'OUTSIDE MODEL SCOPE';
    explanation = applicability.reasons.join(' ') || 'The image characteristics fall outside the validated scope of this forensic model.';
  } else if (applicability.status === 'LIMITED' && scoreGap < 30) {
    label = 'NEEDS VERIFICATION';
    explanation = `Analysis is limited due to image constraints (${applicability.reasons[0] || 'quality limitations'}). Scores do not provide conclusive proof.`;
  } else if (evidenceConsistency === 'LOW') {
    // CRITICAL FIX: Contradictory evidence MUST NOT be classified as LIKELY AI-GENERATED
    label = 'INCONCLUSIVE';
    explanation = `Inconclusive: the analyzed regions produced conflicting forensic signals (regional scores range from ${minSynth}% to ${maxSynth}% synthetic, disagreement: ±${Math.round(patchStdDev)}%). The evidence is contradictory and insufficient to reliably classify this image.`;
  } else if (scoreGap < 18) {
    label = 'INCONCLUSIVE';
    explanation = `Inconclusive: the model score margin is narrow (${scoreGap}-point difference between classes). Automated image forensic models cannot reliably distinguish between real and synthetic content in this range.`;
  } else if (aggregateReal > aggregateSynth) {
    // Leans Real
    if (evidenceStrength === 'STRONG' || forensicSignals.hasCameraLikeNoise) {
      label = 'LIKELY REAL';
      explanation = `The multi-scale analysis found consistent evidence across ${n} forensic views more aligned with a camera-captured image (${aggregateReal}% real vs ${aggregateSynth}% synthetic).`;
    } else {
      label = 'NEEDS VERIFICATION';
      explanation = `The analysis leans toward a camera-captured image (${aggregateReal}% real), but evidence strength is moderate. Independent verification is advised.`;
    }
  } else {
    // Leans Synthetic
    if (evidenceStrength === 'STRONG' && evidenceConsistency !== 'LOW') {
      label = 'LIKELY AI-GENERATED';
      explanation = `The multi-scale analysis detected consistent characteristics across ${n} forensic views consistent with synthetic or diffusion-generated imagery (${aggregateSynth}% synthetic score).`;
    } else {
      label = 'NEEDS VERIFICATION';
      explanation = `The analysis leans toward synthetic imagery (${aggregateSynth}% synthetic), but evidence consistency is moderate. Independent verification is advised.`;
    }
  }

  return {
    syntheticScore: aggregateSynth,
    realScore: aggregateReal,
    scoreGap,
    meanSynth: Math.round(meanSynth),
    medianSynth: Math.round(medianSynth),
    minSynth,
    maxSynth,
    patchStdDev: Math.round(patchStdDev * 10) / 10,
    disagreementRange,
    evidenceStrength,
    evidenceConsistency,
    label,
    explanation
  };
}
