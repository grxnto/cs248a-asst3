implementing "../brdf.slang";

import utils;

public struct LambertianBRDF
{
    public float3 albedo;
    public bool useCosineWeightedSampling;

    public float3 evaluate()
    {
        float3 fr = this.albedo;

        fr = this.albedo * INV_PI;

        return fr;
    }

    public float3 sample(float3 rayOut, float3 normal, float2 randomEpsilon)
    {
        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);

        // HINT: First convert the reflected ray direction to the local space,
        // then compute the incident ray direction in the local space,
        // and finally convert it to the world space.

        if (this.useCosineWeightedSampling)
        {
            // TODO: Student implementation starts here.

            sampleDirection = sampleDirection;

            // TODO: Student implementation ends here.
        }
        else // uniform sampling, use uniform sampling by default
        {
            // TODO: Student implementation starts here.

            float3 localRayOut = mul(rayOut, worldToLocalMatrix); // world to local

            // spherical coordinate sampling
            // since local normal = (0,1,0), hemisphere is y >= 0
            float y = 1- randomEpsilon.x;
            float r = sqrt(1 - y * y);
            float theta = 2 * PI * randomEpsilon.y;

            float x = r * cos(theta);
            float z = r * sin(theta);

            float3 localSampleDirection = float3(x, y, z);

            sampleDirection = mul(localToWorldMatrix, localSampleDirection); // local to world

            // TODO: Student implementation ends here.
        }

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        float EPSILON = 1e-6;
        float pdf = all(abs(rayIn - rayOut) < EPSILON) ? 1.0 : 0.0;

        if (this.useCosineWeightedSampling)
        {
            // TODO: Student implementation starts here.
            
            pdf = pdf;

            // TODO: Student implementation ends here.
        }
        else // uniform sampling, use uniform sampling by default
        {
            // TODO: Student implementation starts here.
            
            pdf = 1 / TWO_PI;

            // TODO: Student implementation ends here.
        }

        return pdf;
    }
}
