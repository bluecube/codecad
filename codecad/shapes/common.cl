float3 quaternion_transform(float4 quaternion, float3 point)
{
    float3 v = as_float3(quaternion);
    return (v * dot(v, point) + cross(v, point) * quaternion.w) * 2 +
           point * (quaternion.w * quaternion.w - dot(v, v));
}

float quaternion_scale(float4 quaternion)
{
    return dot(quaternion, quaternion);
}

// Calculate intersection gradient and distance given two gradient/distance pairs
// with perpendicular gradients
float4 perpendicular_intersection(float4 input1, float4 input2)
{
    if (input1.w > 0 && input2.w > 0)
    {
        float dist = hypot(input1.w, input2.w);

        float3 gradient1 = as_float3(input1);
        float3 gradient2 = as_float3(input2);

        float m1 = input1.w / dist;
        float m2 = input2.w / dist;

        float3 outGradient = gradient1 * m1 + gradient2 * m2;

        float4 out = as_float4(outGradient);
        out.w = dist;

        return out;
    }
    else if (input1.w > input2.w)
        return input1;
    else
        return input2;
}

float4 slab_x(float halfSize, float4 point) {
    return (float4)(copysign(1.0f, point.x), 0, 0, fabs(point.x) - halfSize);
}

float4 slab_y(float halfSize, float4 point) {
    return (float4)(0, copysign(1.0f, point.y), 0, fabs(point.y) - halfSize);
}

float4 slab_z(float halfSize, float4 point) {
    return (float4)(0, 0, copysign(1.0f, point.z), fabs(point.z) - halfSize);
}

float4 rounded_union(float r, float4 obj1, float4 obj2) {

    if (r >= 0)
    {
        float cos_alpha = dot(as_float3(obj1), as_float3(obj2));
        float x1 = r - obj1.w;
        float x2 = r - obj2.w;

        if (cos_alpha * x1 < x2 && cos_alpha * x2 < x1)
        {
            float d = r - sqrt((x1 * x1 + x2 * x2 - 2 * cos_alpha * x1 * x2) / (1 - cos_alpha * cos_alpha));
            return (float4)(0, 0, 0, d); // TODO: Gradient
        }
    }

    if (obj1.w < obj2.w)
        return obj1;
    else
        return obj2;
}

void union_op(float r, float4 obj1, float4 obj2, float4* output) {
    *output = rounded_union(r, obj1, obj2);
}

void intersection_op(float r, float4 obj1, float4 obj2, float4* output) {
    *output = -rounded_union(r, -obj1, -obj2);
}

void subtraction_op(float r, float4 obj1, float4 obj2, float4* output) {
    *output = -rounded_union(r, -obj1, obj2);
}

void transformation_to_op(float qx, float qy, float qz, float qw,
                          float ox, float oy, float oz,
                          float4 point, float4* output) {
    float4 quaternion = (float4)(qx, qy, qz, qw);
    float3 offset = (float3)(ox, oy, oz);

    float3 transformed = quaternion_transform(quaternion, as_float3(point)) + offset;

    *output = as_float4(transformed);
}

void transformation_from_op(float qx, float qy, float qz, float qw,
                            float4 input, float4* output) {
    float4 quaternion = (float4)(qx, qy, qz, qw);

    float scale = quaternion_scale(quaternion);
    float3 transformed = quaternion_transform(quaternion, as_float3(input));

    *output = as_float4(transformed / scale);
    output->w = input.w * scale;
}

void offset_op(float distance, float4 input, float4* output) {
    *output = input - distance;
}

void shell_op(float halfThickness, float4 input, float4* output) {
    *output = fabs(input) - halfThickness;
}

// vim: filetype=c
