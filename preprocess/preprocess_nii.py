import os
import numpy as np
import SimpleITK as sitk


ori_path = '/mnt/data/ssl/data/RICORD/save/RICORD_nii'
save_path = '/mnt/data/ssl/data/RICORD/save/RICORD_nii_resize'


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkLinear):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled


def processing(sub_path_name, i_files):
    img_path = os.path.join(ori_path, sub_path_name, i_files)
    imageITK = sitk.ReadImage(img_path)
    imageITK = resize_image_itk(imageITK, (128, 128, 64))

    target_spacing = imageITK.GetSpacing()
    ori_origin = imageITK.GetOrigin()
    ori_direction = imageITK.GetDirection()
    image = sitk.GetArrayFromImage(imageITK)

    # save
    if not os.path.exists(os.path.join(save_path, sub_path_name)):
        os.makedirs(os.path.join(save_path, sub_path_name))
    saveITK = sitk.GetImageFromArray(image)
    saveITK.SetSpacing(target_spacing)
    saveITK.SetOrigin(ori_origin)
    saveITK.SetDirection(ori_direction)
    sitk.WriteImage(saveITK, os.path.join(save_path, sub_path_name, i_files))


sub_path = ["MIDRC-RICORD-1A", "MIDRC-RICORD-1B"]
for sub_path_name in sub_path:
    for name in os.listdir(os.path.join(ori_path, sub_path_name)):

        if name.endswith("nii.gz"):
            # read img
            print("Processing %s" % (name))
            print("saving at %s" % (os.path.join(save_path, sub_path_name, name)))
            processing(sub_path_name, name)
