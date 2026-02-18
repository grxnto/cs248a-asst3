implementing "../renderer.slang";

import math;
import texture;

public static uint MAX_STEPS = 256;
public static float EPSILON = 1e-6;

public static float STEP_SIZE = 0.01;
public static float DENSITY_SCALE = 5.0;

/**
 * Perform ray marching through the volume and accumulate color.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param tMax The maximum t value for ray marching (usually the nearest primitive intersection t).
 * @param primColor The color of the nearest primitive intersection.
 * @return The accumulated color after ray marching through the volume.
 */
public float4 volumeSample(float2 uv, RendererUniform uniforms, float tMax, float4 primColor)
{
    // TODO: Student implementation starts here.
    Ray ray = uniforms.camera.generateRay(uv);
    float4 o = mul(uniforms.volume.invModelMatrix, float4(ray.origin, 1.0));
    float4 d = mul(uniforms.volume.invModelMatrix, float4(ray.direction, 0.0));
    ray.origin = o.xyz;
    ray.direction = normalize(d.xyz);

    // Call shared accumulation function
    return accumulateColor<1024>(
        ray,
        uniforms.volume.bound,
        MAX_STEPS,
        STEP_SIZE,
        DENSITY_SCALE,
        primColor,
        uniforms.volume.tex
    );
    // TODO: Student implementation ends here.
}

/**
 * Backward pass for volumetric ray marching at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void volumeSampleBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    // TODO: Student implementation starts here.
    // Reconstruct ray (same as forward)
    Ray ray = uniforms.camera.generateRay(uv);
    float4 o = mul(uniforms.volume.invModelMatrix, float4(ray.origin, 1.0));
    float4 d = mul(uniforms.volume.invModelMatrix, float4(ray.direction, 0.0));
    ray.origin = o.xyz;
    ray.direction = normalize(d.xyz);

    DifferentialPtrPair<DiffTexture3D<float, 4>> volPair = 
    DifferentialPtrPair<DiffTexture3D<float, 4>>(
        uniforms.volume.tex, 
        uniforms.volume.dTex
    );
    // Call backward of accumulateColor
    bwd_diff(accumulateColor<256>)(
        ray,
        uniforms.volume.bound,
        MAX_STEPS,
        STEP_SIZE,
        DENSITY_SCALE,
        uniforms.ambientColor, 
        volPair,
        outGrad
    );


    // TODO: Student implementation ends here.
}

/**
 * Ray march through the volume and accumulate color.
 * @param ray The ray in local space.
 * @param volumeBound The bounding box of the volume in local space.
 * @param maxSteps The maximum number of marching steps.
 * @param stepSize The size of each marching step.
 * @param densityScale The scale factor for the volume density (alpha channel).
 * @param backgroundColor The background color behind the volume
                          (this can be the ambient color if there's no primitive behind the volume, or the color of the primitive hit if there's one).
 * @param data The 3D texture storing the volume data (rgba).
 * @return The accumulated color after ray marching through the volume.
 */
[Differentiable]
public float4 accumulateColor<let N : uint>(
    no_diff Ray ray,
    no_diff BoundingBox volumeBound,
    no_diff uint maxSteps,
    no_diff float stepSize,
    no_diff float densityScale,
    no_diff float4 backgroundColor,
    DiffTexture3D<float, 4> data)
{
    // TODO: Student implementation starts here.
    // Intersect ray with volume
    Optional<float2> tBox = volumeBound.hit(ray);
    if (!tBox.hasValue) {
        return backgroundColor;
    }

    float t = tBox.value.x;
    float tEnd = tBox.value.y;
    float3 color = float3(0.0, 0.0, 0.0);  // accumulated
    float transmittance = 1.0; // tracks how much background survives all previous absorption
    uint steps = 0;

    [MaxIters(256)]
    for (uint i = 0; i < N; i++) {
        if (t >= tEnd) break;
        float3 p = ray.origin + t * ray.direction;
        float3 uvw = (p - volumeBound.pMin) / (volumeBound.pMax - volumeBound.pMin);

        // Sample volume
        float4 sample = samplePoint3D(data, uvw);
        // a = density * density_scale * step_size
        float alpha = sample.w * densityScale * stepSize;
        // color = (1.0 - alpha) * color + alpha * sample.xyz;
        // similar to lecture
        color += transmittance * alpha * sample.xyz;
        transmittance *= (1.0 - alpha);

        t += stepSize;
        steps++;
    }

    return float4(color, 1.0 - transmittance) + (transmittance * backgroundColor);

    // TODO: Student implementation ends here.
}
