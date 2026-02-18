implementing "../primitive.slang";
import math;

public struct Triangle : IBoundingBox, IRayCastable
{
    public float3[3] vertices;
    public float3[3] normals;
    public float2[3] uvs;
    public float3[3] colors;
    public uint materialId;
    public uint offset;

    public BoundingBox boundingBox()
    {
        return BoundingBox(float3(0.0), float3(0.0));
    }

    /* Returns the average color of the triangle. */
    public float3 averageColor()
    {
        return (this.colors[0] + this.colors[1] + this.colors[2]) / 3.0;
    }

    /**
     * Calculates the barycentric coordinates of a point on the triangle.
     * @param p The point on the triangle to calculate the barycentric coordinates for.
     * @return The barycentric coordinates of the point.
     */
    public float3 calculateBarycentricCoord(float3 p)
    {
        float3 a = vertices[0];
        float3 b = vertices[1];
        float3 c = vertices[2];

        float3 v0 = b - a;
        float3 v1 = c - a;
        float3 v2 = p - a;

        float d00 = dot(v0, v0);
        float d01 = dot(v0, v1);
        float d11 = dot(v1, v1);
        float d20 = dot(v2, v0);
        float d21 = dot(v2, v1);

        float denom = d00 * d11 - d01 * d01;

        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0 - v - w;

        return float3(u, v, w);

        // TODO: Student implementation ends here.
    }

    public float2 getUV(float3 barycentricCoord)
    {
        float2 uv = float2(0.0);

        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        uv = u * this.uvs[0] + v * this.uvs[1] + w * this.uvs[2];
        
        return uv;
    }

    public float3 getColor(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 color = u * this.colors[0] + v * this.colors[1] + w * this.colors[2];
        return color;
    }

    public float3 getNormal(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 normal = u * this.normals[0] + v * this.normals[1] + w * this.normals[2];
        return normalize(normal);
    }

    /**
     * Test ray-triangle intersection.
     * @param ray The ray to test against the triangle.
     * @return An Optional containing RayHitResult if the ray hits the triangle, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray)
    {
        // TODO: Student implementation starts here.
        float EPSILON = 1e-3;
        float EDGE_EPS = 1e-3;
        float3 normal = normalize(cross((vertices[1] - vertices[0]), (vertices[2] - vertices[0])));

        if (abs(dot(normal, normalize(ray.direction))) < EPSILON) {
            // Ray is nearly parallel to triangle plane
            return Optional<RayHitResult>();
        }

        // Intersecting point on triangle using equation from lecture
        float t = (dot(normal, vertices[0]) - dot(normal, ray.origin)) / dot(normal, normalize(ray.direction));
        if (t <= EPSILON) {
            return Optional<RayHitResult>();
        }
        // compute intersection point of the plane
        float3 x = ray.origin + t * ray.direction;
        
        // Normals of the plane made with an edge of the triangle and normal of triangle plane
        float3 v0 = vertices[0];
        float3 v1 = vertices[1];
        float3 v2 = vertices[2];
        float3 edge0 = v1 - v0;
        float3 edge1 = v2 - v1;
        float3 edge2 = v0 - v2;
        float3 c0 = cross(edge0, x - v0);
        float3 c1 = cross(edge1, x - v1);
        float3 c2 = cross(edge2, x - v2);

        // Dot our point minus vertice vector with the plane normal to determine which side. Plane
        // normals point outward, so it must be less than or equal to zero.
        if (dot(normal, c0) >= -EDGE_EPS && dot(normal, c1) >= -EDGE_EPS && dot(normal, c2) >= -EDGE_EPS) {
            RayHitResult hitResult;
            hitResult.normal = normal;
            hitResult.t = t;
            return Optional<RayHitResult>(hitResult);
        }
        return Optional<RayHitResult>();

        // TODO: Student implementation ends here.
    }

    public uint getMaterialId()
    {
        return this.materialId + this.offset;
    }
}
