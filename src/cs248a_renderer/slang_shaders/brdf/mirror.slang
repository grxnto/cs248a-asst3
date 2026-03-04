implementing "../brdf.slang";

public struct MirrorBRDF
{
    public float3 albedo;

    public float3 evaluate()
    {
        return albedo;
    }

    public float3 sample(float3 rayOut, float3 normal)
    {
        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);

        // HINT: First convert the reflected ray direction to the local space,
        // then compute the incident ray direction in the local space,
        // and finally convert it to the world space.

        // TODO: Student implementation starts here.
        float3 localRayOut = mul(worldToLocalMatrix, rayOut); // world to local
        float3 localNormal = float3(0.0, 1.0, 0.0);
        float3 localIn = reflect(-localRayOut, localNormal); 
        return normalize(mul(localToWorldMatrix, localIn)); // convert back to world space
        // TODO: Student implementation ends here.
        
    }
    

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        const float EPSILON = 1e-6;
        float pdf = all(abs(rayIn - rayOut) < EPSILON) ? 1.0 : 0.0;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);

        // TODO: Student implementation starts here.
        float3 reflected = -rayIn + 2.0 * dot(rayIn, normal) * normal;
        pdf = all(abs(rayOut - reflected) < EPSILON) ? 1.0 : 0.0;
        // TODO: Student implementation ends here.

        return pdf;
    }
}
