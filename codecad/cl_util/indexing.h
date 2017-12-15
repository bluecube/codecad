#ifndef INDEXING_H
#define INDEXING_H

#define INDEX3(sx, sy, x, y, z) ((x) + (sx) * ((y) + (sy) * (z)))
#define INDEX2(sx, x, y) ((x) + (sx) * (y))
#define INDEX3_G(x, y, z) INDEX3(get_global_size(0), get_global_size(1), (x), (y), (z))
#define INDEX2_G(x, y) INDEX2(get_global_size(0), (x), (y))
#define INDEX3_GG INDEX3_G(get_global_id(0), get_global_id(1), get_global_id(2))
#define INDEX2_GG INDEX2_G(get_global_id(0), get_global_id(1))

#endif //INDEXING_H
