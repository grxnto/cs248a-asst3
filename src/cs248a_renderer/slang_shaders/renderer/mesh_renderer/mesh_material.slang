implementing "../renderer.slang";

import utils;

public float2 getShiftedTexUV(float2 uv, RendererUniform uniforms, Triangle triangle, float2 delta, float2 tex_uv)
{
    float2 shifted_uv = uv + delta;
    Ray ray = uniforms.camera.generateRay(shifted_uv);
    Optional<RayHitResult> hitResult = triangle.hit(ray);
    float2 result = tex_uv;
    if (hitResult.hasValue)
    {
        float3 hitPoint = ray.origin + hitResult.value.t * ray.direction;
        float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
        return triangle.getUV(barycentricCoord);
    }

    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult_dx = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult_dx.hasValue)
    {
        Triangle hitTriangle = rayMeshIntersectionResult_dx.value.hitTriangle;
        if (hitTriangle.getMaterialId() == triangle.getMaterialId())
        {
            float3 hitPoint = ray.origin + rayMeshIntersectionResult_dx.value.closestHit.t * ray.direction;
            float3 barycentricCoord = hitTriangle.calculateBarycentricCoord(hitPoint);
            return hitTriangle.getUV(barycentricCoord);
        }
    }

    return tex_uv;
}

public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    float level = 0.0;

    // TODO: Student implementation starts here.
    
    float eps = 0.5 / float(uniforms.camera.canvasSize.y);
    float2 dx = float2(eps, 0.0);
    float2 dy = float2(0.0, eps);

    // generate rays
    Ray ray_px = uniforms.camera.generateRay(uv + dx);
    Ray ray_nx = uniforms.camera.generateRay(uv - dx);
    Ray ray_py = uniforms.camera.generateRay(uv + dy);
    Ray ray_ny = uniforms.camera.generateRay(uv - dy);

    // intersect rays
    Optional<RayMeshIntersectionResult> hit_px = rayMeshIntersection(ray_px, uniforms);
    Optional<RayMeshIntersectionResult> hit_nx = rayMeshIntersection(ray_nx, uniforms);
    Optional<RayMeshIntersectionResult> hit_py = rayMeshIntersection(ray_py, uniforms);
    Optional<RayMeshIntersectionResult> hit_ny = rayMeshIntersection(ray_ny, uniforms);

    // get texture uvs
    float2 tex_uvpx = hit_px.hasValue  ? triangle.getUV(triangle.calculateBarycentricCoord(ray_px.origin + hit_px.value.closestHit.t * ray_px.direction)) : tex_uv;
    float2 tex_uvnx = hit_nx.hasValue  ? triangle.getUV(triangle.calculateBarycentricCoord(ray_nx.origin + hit_nx.value.closestHit.t * ray_nx.direction)) : tex_uv;
    float2 tex_uvpy = hit_py.hasValue  ? triangle.getUV(triangle.calculateBarycentricCoord(ray_py.origin + hit_py.value.closestHit.t * ray_py.direction)) : tex_uv;
    float2 tex_uvny = hit_ny.hasValue  ? triangle.getUV(triangle.calculateBarycentricCoord(ray_ny.origin + hit_ny.value.closestHit.t * ray_ny.direction)) : tex_uv;


    // texture size
    PhysicsBasedMaterial mat = uniforms.physicsBasedMaterialBuf[triangle.materialId];
    float w = (float)mat.albedo.mipmap[0].size.x;
    float h = (float)mat.albedo.mipmap[0].size.y;


    float2 duv_dx = (tex_uvpx - tex_uvnx) * w;
    float2 duv_dy = (tex_uvpy - tex_uvny) * h;
    float max_sq = max(dot(duv_dx, duv_dx), dot(duv_dy, duv_dy));

    return max(0.0, 0.5 * log2(max_sq));

    // TODO: Student implementation ends here.

    return level;
}

public PhysicsBasedParameters getHitPointParameters(float2 uv, RendererUniform uniforms, Triangle triangle, float3 normal, float3 barycentricCoord, float3 hitPoint)
{
    float2 tex_uv = triangle.getUV(barycentricCoord);
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    if (uniforms.visualizeBarycentricCoords)
    {
        parameters.albedo = barycentricCoord;
    }
    if (uniforms.visualizeTexUV)
    {
        parameters.albedo = float3(tex_uv, 0.0);
    }
    return parameters;
}
