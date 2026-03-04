implementing "../brdf.slang";

public struct GlassBRDF
{
    public float3 albedo;
    public float ior;

    public float3 evaluate()
    {
        return albedo;
    }

    public float3 sample(float3 rayOut, float3 normal)
    {   
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);
        // TODO: Student implementation starts here.
        float3 localRayOut = mul(worldToLocalMatrix, rayOut); // world to local
        float3 localNormal = float3(0.0, 1.0, 0.0); // normal in local space


        // determine if the ray is entering. if local ray y-component is positive, it is above the surface.
        bool entering = localRayOut.y > 0.0;
        float n1;
        float n2;
        if (entering) {
            n1 = 1.0; // medium from
            n2 = ior; // medium exiting
        } else {
            n1 = ior; // medium from
            n2 = 1.0; // medium exiting
            localNormal = -localNormal; // flip normal to point against the ray
        }
        float eta = n1 / n2;
        
        // compute the refracted ray direction using Snell's law
        // this is cos^2(theta) of the refracted ray
        float k = 1.0 - eta * eta * (1.0 - dot(localRayOut, localNormal) * dot(localRayOut, localNormal));

        float3 localIn;
        if (k < 0.0) {
            // Total internal reflection
            localIn = reflect(-localRayOut, localNormal); 
        } else {
            // Refraction
            localIn = refract(-localRayOut, localNormal, eta);
        }

        return normalize(mul(localToWorldMatrix, localIn));

        // TODO: Student implementation ends here.
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        // For dielectric materials, the PDF is 1.0 for the single direction
        // (either refracted or reflected due to total internal reflection)
        return 1.0;
    }
}
