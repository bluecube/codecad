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

float4 slab_x(float halfSize, float4 point)
{
    return (float4)(copysign(1.0f, point.x), 0, 0, fabs(point.x) - halfSize);
}

float4 slab_y(float halfSize, float4 point)
{
    return (float4)(0, copysign(1.0f, point.y), 0, fabs(point.y) - halfSize);
}

float4 slab_z(float halfSize, float4 point)
{
    return (float4)(0, 0, copysign(1.0f, point.z), fabs(point.z) - halfSize);
}

uchar union_op(__constant float* params, float4* output, float4 obj1, float4 obj2) {
    if (obj1.w < obj2.w)
        *output = obj1;
    else
        *output = obj2;
    return 1; // TODO: Actually use the radius

    /*
    @staticmethod
    def distance2(r, s1, s2, point):
        epsilon = min(s1.bounding_box().size().min(),
                      s2.bounding_box().size().min()) / 10000;

        d1 = s1.distance(point)
        d2 = s2.distance(point)
        x1 = r - d1
        x2 = r - d2

        # epsilon * gradient(s1)(point)
        g1 = util.Vector(s1.distance(point + util.Vector(epsilon, 0, 0)) - d1,
                         s1.distance(point + util.Vector(0, epsilon, 0)) - d1,
                         s1.distance(point + util.Vector(0, 0, epsilon)) - d1)

        cos_alpha = abs((s2.distance(point + g1) - d2) / epsilon)

        dist_to_rounding = r - util.sqrt((x1 * x1 + x2 * x2 - 2 * cos_alpha * x1 * x2) / (1 - cos_alpha * cos_alpha))

        cond1 = (cos_alpha * x1 < x2)
        cond2 = (cos_alpha * x2 < x1)

        return util.switch(cond1 & cond2, dist_to_rounding, util.minimum(d1, d2))
    */
}

uchar intersection_op(__constant float* params, float4* output, float4 obj1, float4 obj2) {
    if (obj1.w > obj2.w)
        *output = obj1;
    else
        *output = obj2;
    return 1; // TODO: Actually use the radius
}

uchar subtraction_op(__constant float* params, float4* output, float4 obj1, float4 obj2) {
    return intersection_op(params, output, obj1, -obj2);
        // "-" here both negates distance and turns around gradient vector
}

uchar transformation_to_op(__constant float* params, float4* output, float4 point, float4 unused) {
    float4 quaternion = vload4(0, params);
    float3 offset = vload3(0, params + 4);

    float3 transformed = quaternion_transform(quaternion, as_float3(point)) + offset;

    *output = as_float4(transformed);

    return 7;
}

uchar transformation_from_op(__constant float* params, float4* output, float4 input, float4 unused) {
    float4 quaternion = vload4(0, params);

    float scale = quaternion_scale(quaternion);
    float3 transformed = quaternion_transform(quaternion, as_float3(input));

    *output = as_float4(transformed / scale);
    output->w = input.w * scale;

    return 4;
}

uchar offset_op(__constant float* params, float4* output, float4 input, float4 unused) {
    float distance = params[0];
    *output = input - distance;
    return 1;
}

uchar shell_op(__constant float* params, float4* output, float4 input, float4 unused) {
    float half_thickness = params[0];
    *output = fabs(input) - half_thickness;
    return 1;
}

// vim: filetype=c
