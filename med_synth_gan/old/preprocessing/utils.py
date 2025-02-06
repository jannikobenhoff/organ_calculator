import nibabel as nib
import numpy as np
import json

def add_label_map_to_nifti(img_in, label_map):
    """
    This will save the information which label in a segmentation mask has which name to the extended header.

    img: nifti image
    label_map: a dictionary with label ids and names | a list of names and a running id will be generated starting at 1

    returns: nifti image
    """
    data = img_in.get_fdata()

    if label_map is None:
        label_map = {idx+1: f"L{val}" for idx, val in enumerate(np.unique(data)[1:])}

    if type(label_map) is not dict:   # can be list or dict_values list
        label_map = {idx+1: val for idx, val in enumerate(label_map)}

    colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,128,0],[255,0,128],[128,255,128],[0,128,255],[128,128,128],[185,170,155]]
    xmlpre = '<?xml version="1.0" encoding="UTF-8"?> <CaretExtension>  <Date><![CDATA[2013-07-14T05:45:09]]></Date>   <VolumeInformation Index="0">   <LabelTable>'

    body = ''
    for label_id, label_name in label_map.items():
        rgb = colors[label_id%len(colors)]
        body += f'<Label Key="{label_id}" Red="{rgb[0]/255}" Green="{rgb[1]/255}" Blue="{rgb[2]/255}" Alpha="1"><![CDATA[{label_name}]]></Label>\n'

    xmlpost = '  </LabelTable>  <StudyMetaDataLinkSet>  </StudyMetaDataLinkSet>  <VolumeType><![CDATA[Label]]></VolumeType>   </VolumeInformation></CaretExtension>'
    xml = xmlpre + "\n" + body + "\n" + xmlpost + "\n              "

    img_in.header.extensions.append(nib.nifti1.Nifti1Extension(0,bytes(xml,'utf-8')))

    return img_in


def save_multilabel_nifti(img, output_path, label_map, nora_project=None):
    """
    img: nifti image
    output_path: the output path
    label_map: a dictionary with label ids and names | a list of names and a running id will be generated starting at 1
    nora_project: if provided the file will be tagged as 'atlas'
    """
    img = add_label_map_to_nifti(img, label_map)
    nib.save(img, output_path)
    if nora_project is not None:
        subprocess.call(f"/opt/nora/src/node/nora -p {nora_project} --add {str(output_path)} --addtag atlas", shell=True)


def load_multilabel_nifti(img_path):
    """
    img_path: path to the image
    returns:
        img: nifti image
        label_map: a dictionary with label ids and names
    """
    import xmltodict
    img = nib.load(img_path)
    ext_header = img.header.extensions[0].get_content()
    ext_header = xmltodict.parse(ext_header)
    ext_header = ext_header["CaretExtension"]["VolumeInformation"]["LabelTable"]["Label"]
    
    # If only one label, ext_header is a dict instead of a list (because of xmltodict.parse()) -> convert to list
    if isinstance(ext_header, dict):
        ext_header = [ext_header]
        
    label_map = {int(e["@Key"]): e["#text"] for e in ext_header}
    return img, label_map  

def create_itksnap_label_file(label_mapping, output_path):
    """
    Creates an ITK-SnAP label description file with unique colors for each label.
    
    Args:
        label_mapping (dict): Dictionary mapping anatomical parts to label numbers
        output_path (str): Path to save the label description file
    """
    import colorsys
    
    def generate_distinct_colors(n):
        colors = []
        golden_ratio_conjugate = 0.618033988749895
        hue = 0
        for _ in range(n):
            hue = (hue + golden_ratio_conjugate) % 1.0
            # Use fixed saturation and value for vibrant colors
            hsv = (hue, 0.9, 0.95)
            rgb = colorsys.hsv_to_rgb(*hsv)
            # Convert to 0-255 range
            colors.append([int(c * 255) for c in rgb])
        return colors
    
    header = """################################################
# ITK-SnAP Label Description File
# File format: 
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields: 
#    IDX:   Zero-based index 
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    IDX:   Label mesh visibility (0 or 1)
#  LABEL:   Label description 
################################################"""

    # Generate unique colors for all labels
    num_labels = max(label_mapping.values())
    colors = generate_distinct_colors(num_labels)

    with open(output_path, 'w') as f:
        # Write header
        f.write(header + '\n')
        
        # Write background label
        f.write('    0     0    0    0        0  0  0    "Clear Label"\n')
        
        # Write labels for each anatomical part
        for part, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
            if idx == 0:  # Skip background
                continue
                
            color = colors[idx-1]  # Use generated color
            label_name = part.replace('_', ' ').title()
            f.write(f'{idx:5d} {color[0]:4d} {color[1]:4d} {color[2]:4d}        1  1  1    "{label_name}"\n')


def txt_to_json(filename):
    """
    Converts a txt file with label names to a json file.
    The txt file should contain one label name per line.
    """

    output = {}
    with open(filename, 'r') as f:
        label_names = f.read().splitlines()

        for label in label_names:
            splitted = label.split(":")
            key = splitted[0]
            value = splitted[1].strip()
            output[value] = int(key)
    
    json.dump(output, open(filename.replace(".txt", ".json"), "w"), indent=2)


if __name__ == "__main__":
    txt_to_json("../data/totalsegmentator_combined/label_mapping.txt")