import os
import shutil
import logging

import tqdm
import imageio


DB_PATH = os.path.join("data", "object_datasets", "shapenet_core_v2")


## PROBLEM: Typo in texture paths
## FIX: Path to textures './images' replaced by '../images'
for short_tmpl_id in tqdm.tqdm(("car/b3ffbbb2e8a5376d4ed9aac513001a31",
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
                                "chair/7ad134826824de98d0bef5e87b92b95e"),
                               desc="Editing image paths in '.mtl'"):
    fpath = os.path.join(DB_PATH, *short_tmpl_id.split('/'), "models", "model_normalized.mtl")
    try:
        with open(fpath) as f:
            mtl = f.read()
        logging.info(f"Renaming texture dir in '{fpath}'")
        mtl = mtl.replace(" ./images", " ../images")
        with open(fpath, 'wt') as f:
            f.write(mtl)
    except FileNotFoundError:
        logging.warning(f"'{fpath}' not found.")
        continue


## PROBLEM: Unnormalized textures path and format ('.tmp')
## FIX: Converted textures to '.png', Moved to '../images', Edited '.mtl' accordingly
for short_tmpl_id in tqdm.tqdm(("car/191f9cd970e5b0cc174ee7ebab9d8065",
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
                                "car/f1a20b5b20366d5378df335b5e4f64d1"),
                               desc="Converting textures to proper format"):
    fpath = os.path.join(DB_PATH, *short_tmpl_id.split('/'), "models", "model_normalized.mtl")
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
                    dst_path = os.path.join(dst_dir, "..", "images", src_name + ".png")
                    logging.info(f"Converting texture from '{rel_path}' to '{dst_path}'")
                    img = imageio.imread(src_path)
                    imageio.imwrite(dst_path, img)
                    f.write(f"{key} ../images/{src_name}.png\n")
                else:
                    f.write(l)
    except FileNotFoundError:
        logging.warning(f"'{fpath}' not found.")
        continue

## PROBLEM: Various unfixable...
## FIX: Remove object from dataset
for short_tmpl_id in tqdm.tqdm((
    ## PB: Missing textures
    "airplane/d583d6f23c590f3ec672ad25c77a396", # '6.jpg'
    "bus/2d44416a2f00fff08fd1fd158db1677c", # '4.jpg'
    "car/fe3dc721f5026196d61b6a34f3fd808c", # '0.jpg'
    "car/ec67edc59aef93d9f5274507f44ab711", # '0.JPG'
    "car/39b307361b650db073a425eed3ac7a0b", # '0.png' '1.PNG'
    "car/8242b114695b68286f522b2bb8ded829", # '0.jpg'
    "car/98a4518ee8e706c94e84ac3ac08acdb2", # '0.jpg'
    "car/66be76d60405c58ae02ca9d4b3cbc724", # '0.jpg' '1.jpg' '2.jpg'
    "car/f11d669c1c57a595cba0d757b1f2aa52", # '0.jpg' '1.jpg' '2.jpg' '3.JPG'
    "car/f6bbb767b1b75ab0c9d22fcb1abe82ed", # '0.jpg' '1.JPG' '2.JPG' '3.JPG'
    "car/61f4cd45f477fc7a48a1f672d5ac8560", # '0.jpg'
    "car/648ceaad362345518a6cf8c6b92417f2", # '0.jpg', '1.jpg'
    "car/e95d4b7aa9617eb05c58fd6a60e080a", # '0.jpg', '1.jpg'
    "car/a1d85821a0666a4d8dc995728b1ad443", # '0.JPG'
    "car/846f4ad1db06d8791e0b067dee925db4", # '0.jpg'
    "car/685f2b388b018ab78cab9eeff9aeaee2", # '0.JPG' '1.JPG' '2.JPG'
    "car/e2ceb9bf23b498dda7431386d9d22644", # '0.jpg'
    "car/d5f1637a5c9479e0185ce5d54f27f6b9", # '0.jpg' '1.jpg'
    "car/a262c2044977b6eb52ab7aae4be20d81", # '0.jpg'
    "car/731efc7a52841a5a59139efcde1fedcb", # '0.jpg'
    "car/85914342038de6f160190e29962cb3e7", # '0.jpg' '1.jpg'
    "chair/b7a1ec97b8f85127493a4a2a112261d3", # '0.jpg'
    "telephone/89d70d3e0c97baaa859b0bef8825325f", # '0.png'
    "vessel/6367d10f3cb043e1cdcba7385a96c2c8", # '2.jpg'
    ## PB: Corrupted texture files?
    "airplane/1e6a71e0cb436a88a3a1394d6e3d2c63",
    "guitar/4275faf3b0234007f03058f020a47ca5",
    ## PB: Missing '.obj'
    "car/806d740ca8fad5c1473f10e6caaeca56",
    "car/2307b51ca7e4a03d30714334794526d4",
    "car/e6c22be1a39c9b62fb403c87929e1167",
    "car/207e69af994efa9330714334794526d4",
    "car/986ed07c18a2e5592a9eb0f146e94477",
    "car/f5bac2b133a979c573397a92d1662ba5",
    "car/d6ee8e0a0b392f98eb96598da750ef34",
    "car/5973afc979049405f63ee8a34069b7c5",
    "car/9fb1d03b22ecac5835da01f298003d56",
    "car/302612708e86efea62d2c237bfbc22ca",
    "car/3ffeec4abd78c945c7c79bdb1d5fe365",
    "car/8070747805908ae62a9eb0f146e94477",
    "car/7aa9619e89baaec6d9b8dfa78596b717",
    "car/4ddef66f32e1902d3448fdcb67fe08ff",
    "car/5bf2d7c2167755a72a9eb0f146e94477",
    "car/3c33f9f8edc558ce77aa0b62eed1492",
    "car/93ce8e230939dfc230714334794526d4",
    "car/407f2811c0fe4e9361c6c61410fc904b",
    "car/ea3f2971f1c125076c4384c3b17a86ea",
    ## PB: Invalid '.obj' file (out of range vertex normal index)
    "car/6188f5bafcd88523215d73d375c978d5",
    ## PB: Empty texture paths '../'
    "chair/2ae70fbab330779e3bff5a09107428a5",
    "chair/a8c0ceb67971d0961b17743c18fb63dc",
    "chair/f3c0ab68f3dab6071b17743c18fb63dc",
    "chair/c70c1a6a0e795669f51f77a6d7299806",
    "clock/5972bc07e59371777bcb070cc655f13a",
    "sofa/191c92adeef9964c14038d588fd1342f",
    ## PB: Empty texture files
    "display/b0952767eeb21b88e2b075a80e28c81b",
    "telephone/a4910da0271b6f213a7e932df8806f9e",
), desc="Removing unfixable objects"):
    cat_name, model_id = short_tmpl_id.split('/')
    dirpath = os.path.join(DB_PATH, cat_name, model_id)
    if os.path.isdir(dirpath):
        logging.info(f"Recursively deleting directory '{dirpath}'")
        shutil.rmtree(dirpath)
        os.remove(dirpath + ".object_config.json")
    else:
        logging.warning(f"'{dirpath}' not found.")
