implementing "../model.slang";
import math;

public struct BVHNode
{
    public BoundingBox bound;
    public int left;
    public int right;
    public int primLeft;
    public int primRight;
    public int depth;

    public __init(BoundingBox bound, int left = -1, int right = -1, int primLeft = -1, int primRight = -1, int depth = -1)
    {
        this.bound = bound;
        this.left = left;
        this.right = right;
        this.primLeft = primLeft;
        this.primRight = primRight;
        this.depth = depth;
    }

    public float2 hit(Ray ray)
    {
        return this.bound.hit(ray);
    }
}

public struct BVH<T>
    where T : IRayCastable
{
    public struct HitResult
    {
        public T primitive;
        public RayHitResult rayHitResult;
    }

    // Slang buffer of BVHNode copied from Python.
    public RWStructuredBuffer<BVHNode> nodes;
    // Maximum number of nodes in the BVH, in this case, it'll be the size of the nodes buffer.
    public uint maxNodes;
    // Slang buffer of primitives represented by the BVH.
    public RWStructuredBuffer<T> primitives;
    // Number of primitives in the primitives buffer.
    public uint numPrimitives;

    public __init(RWStructuredBuffer<BVHNode> nodes, RWStructuredBuffer<T> primitives, uint numPrimitives, uint maxNodes)
    {
        this.nodes = nodes;
        this.primitives = primitives;
        this.numPrimitives = numPrimitives;
        this.maxNodes = maxNodes;
    }

    struct HitStack<int N>
    {
        uint[N] nodeIndices;
        uint top;

        __init()
        {
            this.top = 0;
        }

        [mutating]
        void push(uint nodeIndex)
        {
            this.nodeIndices[this.top] = nodeIndex;
            this.top += 1;
        }

        [mutating]
        uint pop()
        {
            this.top -= 1;
            return this.nodeIndices[this.top];
        }

        bool isEmpty()
        {
            return this.top == 0;
        }

        uint size()
        {
            return this.top;
        }
    }

    // BVH traversal to find the closest hit.
    public Optional<HitResult> hit(Ray ray)
    {
        // Initialize traversal.
        Optional<HitResult> hitResult = Optional<HitResult>();
        HitStack<64> stack;

        // TODO: Student implementation starts here.

        // start at root and closest is infinity
        stack.push(0);
        float closest = float.maxValue;


        // while stack is not empty, 
        while(!stack.isEmpty()) {
            // get node at the top
            uint i = stack.pop();
            BVHNode node = this.nodes[i];
            if (i >= this.maxNodes) {
                continue;
            }

            // get a hit of the node's bounding box, if there is a miss or farther than the closest then skip
            float2 bb = node.bound.hit(ray);
            if (bb.x > bb.y || bb.x > closest) {
                continue;
            }  

            // if the node is a leaf node, intersect with all primitives in leaf node and update closest one
            if (node.left == -1 && node.right == -1) {
                for (int i = node.primLeft; i < node.primRight; i++) {
                    Optional<RayHitResult> primitive_hit = this.primitives[i].hit(ray);
                    if (primitive_hit.hasValue && primitive_hit.value.t < closest) {
                        // if closer, 
                        closest = primitive_hit.value.t;
                        
                        HitResult result;
                        result.primitive = this.primitives[i];
                        result.rayHitResult = primitive_hit.value;
                        hitResult = Optional<HitResult>(result);
                    }
                }
            }
            else {
                // push children onto stack
                if (node.left != -1)
                    stack.push((uint)node.left);
                if (node.right != -1)
                    stack.push((uint)node.right);
            }

        }
        // TODO: Student implementation ends here.

        return hitResult;
    }
}
