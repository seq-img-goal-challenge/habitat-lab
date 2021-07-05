import os
import imageio

## PROBLEM: Typo in texture paths
## FIX: Path to textures './images' replaced by '../images'
for short_tmpl_id in ("car/b3ffbbb2e8a5376d4ed9aac513001a31",
                      "car/2854a948ff81145d2d7d789814cae761",
                      "car/15fcfe91d44c0e15e5c9256f048d92d2",
                      "car/373cf6c8f79376482d7d789814cae761",
                      "car/1c66dbe15a6be89a7bfe055aeab68431",
                      "car/db14f415a203403e2d7d789814cae761",
                      "car/bb7fec347b2b57498747b160cf027f1",
                      "car/662cd6478fa3dd482d7d789814cae761",
                      "car/558404e6c17c58997302a5e36ce363de",
                      "chair/941720989a7af0248b500dd30d6dfd0",
                      "chair/482afdc2ddc5546f764d42eddc669b23",
                      "chair/2b90701386f1813052db1dda4adf0a0c",
                      "chair/7ad134826824de98d0bef5e87b92b95e"):
    fpath = os.path.join("data", "object_datasets", "shapenet_core_v2",
                         *short_tmpl_id.split('/'), "models", "model_normalized.mtl")
    try:
        with open(fpath) as f:
            mtl = f.read()
        mtl = mtl.replace(" ./images", " ../images")
        with open(fpath, 'wt') as f:
            f.write(mtl)
    except FileNotFoundError:
        continue


## PROBLEM: Unnormalized textures path and format ('.tmp')
## FIX: Converted textures to '.png', Moved to '../images', Edited '.mtl' accordingly
for short_tmpl_id in ("car/191f9cd970e5b0cc174ee7ebab9d8065",
                      "car/324434f8eea2839bf63ee8a34069b7c5",
                      "car/9f69ac0aaab969682a9eb0f146e94477",
                      "car/8bbbfdbec9251733ace5721ccacba16",
                      "car/7c7e5b4fa56d8ed654b40bc735c6fdf6",
                      "car/355e7a7bde7d43c1ace5721ccacba16",
                      "car/baa424fcf0badeedd485372bb746f3c7",
                      "car/db86af2f9e7669559ea4d0696ca601de",
                      "car/631aae18861692042026875442db8f4d",
                      "car/c07bbf672b56b02aafe1d4530f4c6e24",
                      "car/330645ba272910524376d2685f42f96f",
                      "car/f378404d31ce9db1afe1d4530f4c6e24",
                      "car/63f6a2c7ee7c667ba0b677182d16c198",
                      "car/8aa9a549372e44143765ee7ffdfef49f",
                      "car/857a3a01bd311511f200a72c9245aee7",
                      "car/3ac664a7486a0bdff200a72c9245aee7",
                      "car/706671ef8c7b5e28a6c2c95b41a5446d",
                      "car/381332377d8aff57573c99f10261e25a",
                      "car/31055873d40dc262c7477eb29831a699",
                      "car/5721c147ce05684d613dc416ee51531e",
                      "car/4e9a489d830e285b59139efcde1fedcb",
                      "car/490812763fa965b8473f10e6caaeca56",
                      "car/1724ae84377e0b9ba6c2c95b41a5446d",
                      "car/4e009085e3905f2159139efcde1fedcb",
                      "car/4e384de22a760ef72e877e82c90c24d",
                      "car/54b89bb4ed5aa492e23d60a1b706b44f",
                      "car/a9e8123feadc58d5983c36827cbbba97",
                      "car/b812523dddd4a34a473f10e6caaeca56",
                      "car/f1b928427f5a3a7fc6d5ebddbbbc39f",
                      "car/f1a20b5b20366d5378df335b5e4f64d1"):
    fpath = os.path.join("data", "object_datasets", "shapenet_core_v2",
                         *short_tmpl_id.split('/'), "models", "model_normalized.mtl")
    try:
        with open(fpath) as f:
            mtl = f.readlines()
        with open(fpath, 'wt') as f:
            for l in mtl:
                if l.startswith("map_"):
                    key, rel_path = l.strip().split(' ', 1)
                    src_name, _ = os.path.splitext(os.path.basename(rel_path))
                    dst_dir = os.path.dirname(fpath)
                    src_path = os.path.join(dst_dir, rel_path)
                    img = imageio.imread(src_path)
                    dst_path = os.path.join(dst_dir, "..", "images", src_name + ".png")
                    imageio.imwrite(dst_path, img)
                    f.write(f"{key} ../images/{src_name}.png\n")
                else:
                    f.write(l)
    except FileNotFoundError:
        continue

