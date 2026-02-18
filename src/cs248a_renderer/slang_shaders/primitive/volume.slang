implementing "../primitive.slang";
import math;
import texture;

// The size of each marching step.
static float STEP_SIZE = 0.005;
// The density threshold to consider a hit.
static float DENSITY_THRESHOLD = 0.25;
// The delta used for finite difference normal estimation.
static float FINITE_DIFF_DELTA = 0.001;

public struct Volume
{
    public BoundingBox bound;
    public SharedTexture3D tex;
    public float4x4 modelMatrix;
    public float4x4 invModelMatrix;

    /**
     * Test ray-volume bounding box intersection.
     * @param ray The ray to test against the volume.
     * @param volumeTexBuf The texture buffer contains all the volumetrically represented models in the scene .
     * @return An Optional containing RayHitResult if the ray hits the volume, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray, SharedTexture3DBuffer<float4> volumeTexBuf)
    {
        // TODO: Student implementation starts here.
        // transform ray into the world space
        Ray local_ray = ray.transform(invModelMatrix);

        // bounding box intersections
        float2 tBox = bound.hit(local_ray);
        float tMin = tBox.x;
        float tMax = tBox.y;

        // return if ray misses the bounding box
        if (tMax < tMin) {
            return Optional<RayHitResult>();
        }
    
        float t = tMin;
        while (t <= tMax) {
            // get point in world space to normalized screen space
            float3 p = local_ray.origin + t * local_ray.direction;
            float3 uvw = getUVW(p);

            // sample from the volume for the density 
            float density = volumeTexBuf.trilinearSample(tex, uvw).w;
            if (density > DENSITY_THRESHOLD) {
                    // estimate normal

                    float gx = volumeTexBuf.trilinearSample(tex, uvw + float3(FINITE_DIFF_DELTA, 0, 0)).w - volumeTexBuf.trilinearSample(tex, uvw - float3(FINITE_DIFF_DELTA, 0, 0)).w;
                    float gy = volumeTexBuf.trilinearSample(tex, uvw + float3(0, FINITE_DIFF_DELTA, 0)).w - volumeTexBuf.trilinearSample(tex, uvw - float3(0, FINITE_DIFF_DELTA, 0)).w;
                    float gz = volumeTexBuf.trilinearSample(tex, uvw + float3(0, 0, FINITE_DIFF_DELTA)).w - volumeTexBuf.trilinearSample(tex, uvw - float3(0, 0, FINITE_DIFF_DELTA)).w;

                    float3 local_normal = -normalize(float3(gx, gy, gz));
                    float3 normal = normalize(mul((float3x3)modelMatrix, local_normal));
                    // Transform normal to world space
                    RayHitResult result;
                    result.t = t;
                    result.normal = normal;

                    return Optional<RayHitResult>(result);
            }
            t += STEP_SIZE;
        }

        return Optional<RayHitResult>();
        // TODO: Student implementation ends here.

    }

    /**
     * Get the uvw coordinates of a point in local space.
     * @param p The point in world space.
     * @return The uvw coordinates in [0, 1].
     */
    public float3 getUVW(float3 p)
    {
        // Compute uvw coordinates in [0, 1].
        float3 minBound = this.bound.pMin;
        float3 maxBound = this.bound.pMax;
        return (p - minBound) / (maxBound - minBound);
    }
};
