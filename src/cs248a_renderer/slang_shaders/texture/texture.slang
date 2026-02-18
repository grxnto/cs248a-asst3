implementing "../texture.slang";

public struct SharedTexture2DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture2D tex, uint2 texelCoord)
    {
        uint2 texSize = tex.size;
        uint offset = tex.offset;
        uint index = texelCoord.y * texSize.x + texelCoord.x;
        return buffer[offset + index];
    }

    public T lerpT(T a, T b, float t) {
        return a + (b - a) * T(t);
    }

    public T pointSample(SharedTexture2D tex, float2 uv)
    {
        // TODO: Student implementation starts here.

        // Map to pixel space
        float2 p = uv * float2(tex.size) - 0.5;

        // Nearest neighbor
        int2 coord = int2(floor(p + 0.5));
        coord = clamp(coord, int2(0,0), int2(tex.size) - 1);

        return getBufferValue(tex, uint2(coord));
        // TODO: Student implementation ends here.
    }

    public T bilinearSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        // Convert UV to pixel space
        float2 pixel = uv * float2(tex.size) - 0.5;

        // Compute integer pixel coordinates and clamp
        int x0 = clamp(int(floor(pixel.x)), 0, int(tex.size.x) - 1);
        int y0 = clamp(int(floor(pixel.y)), 0, int(tex.size.y) - 1);
        int x1 = min(x0 + 1, int(tex.size.x) - 1);
        int y1 = min(y0 + 1, int(tex.size.y) - 1);

        // Fractional part
        float2 t = pixel - float2(x0, y0);

        // Sample neighbors
        T c00 = getBufferValue(tex, uint2(x0, y0));
        T c10 = getBufferValue(tex, uint2(x1, y0));
        T c01 = getBufferValue(tex, uint2(x0, y1));
        T c11 = getBufferValue(tex, uint2(x1, y1));

        // Bilinear interpolation
        return lerpT(lerpT(c00, c10, t.x), lerpT(c01, c11, t.x), t.y);
        // TODO: Student implementation ends here.
    }

    public T trilinearSample(SharedTexture2D[] mipmapLevels, float2 uv, float d, uint totalLevels)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        d = clamp(d, 0.0, float(totalLevels - 1));

        uint level0 = uint(floor(d));
        uint level1 = min(level0 + 1, totalLevels - 1);
        float factor = d - float(level0);

        // bilinear sample each level 
        T sample0 = bilinearSample(mipmapLevels[level0], uv);
        T sample1 = bilinearSample(mipmapLevels[level1], uv);

        // interpolate between levels
        outV = lerpT(sample0, sample1, factor);

        // TODO: Student implementation ends here.
        return outV;
    }
}

public struct SharedTexture2D
{
    public uint2 size;
    public uint offset;
}

public struct SharedTexture3DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture3D tex, uint3 voxelCoord)
    {
        uint3 texSize = tex.size;
        uint offset = tex.offset;
        uint index = voxelCoord.z * texSize.y * texSize.x +
                     voxelCoord.y * texSize.x +
                     voxelCoord.x;
        return buffer[offset + index];
    }
    
    public T lerpT(T a, T b, float t) {
        return a + (b - a) * T(t);
    }

    /**
     * Sample the texture at the given uvw coordinates using nearest neighbor interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     */
    public T pointSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        return getBufferValue(tex, uint3(round(uvw * tex.size)));

        // TODO: Student implementation ends here.
    }

    /**
     * Sample the texture at the given uvw coordinates using trilinear interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     **/
    public T trilinearSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        // uvw to voxel space and find voxel cube (points surrounding it)
        float3 voxel = uvw * float3(tex.size);
        int3 min = int3(0, 0, 0);
        int3 max = int3(tex.size) - 1;
        int3 start = int3(floor(voxel));
        float3 t = voxel - float3(start);
        
        // Get information at each point surrounding uvw of the geometry sample
        T c000 = getBufferValue(tex, uint3(start.x, start.y, start.z));
        T c100 = getBufferValue(tex, uint3(start.x + 1, start.y, start.z));
        T c010 = getBufferValue(tex, uint3(start.x, start.y + 1, start.z));
        T c110 = getBufferValue(tex, uint3(start.x + 1, start.y + 1, start.z));

        T c001 = getBufferValue(tex, uint3(start.x, start.y, start.z + 1));
        T c101 = getBufferValue(tex, uint3(start.x + 1, start.y, start.z + 1));
        T c011 = getBufferValue(tex, uint3(start.x, start.y + 1, start.z + 1));
        T c111 = getBufferValue(tex, uint3(start.x + 1, start.y + 1, start.z + 1));

        // using its floating coordinate minus the start of the voxel, we find the weighted average with modified lerp function
        // first interpolate on x, one for each face (around voxel left right) (bottom to top) (near and far)
        return lerpT(lerpT(lerpT(c000, c100, t.x), lerpT(c010, c110, t.x), t.y), lerpT(lerpT(c001, c101, t.x), lerpT(c011, c111, t.x), t.y), t.z);


        // TODO: Student implementation ends here.
    }
}

public struct SharedTexture3D
{
    public uint3 size;
    public uint offset;
}

