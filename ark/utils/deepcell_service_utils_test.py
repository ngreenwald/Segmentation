import pathlib
import os
import tempfile
from zipfile import ZipFile
import pytest

from ark.utils.deepcell_service_utils import create_deepcell_output


def mocked_run_deepcell(input_dir, output_dir, host, job_type):
    pathlib.Path(os.path.join(output_dir, 'example_point1_feature_0.tif')).touch()
    pathlib.Path(os.path.join(output_dir, 'example_point2_feature_0.tif')).touch()

    zip_path = os.path.join(output_dir, 'example_output.zip')
    with ZipFile(zip_path, 'w') as zipObj:
        for i in [1, 2]:
            filename = os.path.join(output_dir, f'example_point{i}_feature_0.tif')
            zipObj.write(filename, os.path.basename(filename))
            os.remove(filename)


def test_create_deepcell_output(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        mocker.patch('ark.utils.deepcell_service_utils.run_deepcell_task', mocked_run_deepcell)

        input_dir = os.path.join(temp_dir, 'input_dir')
        os.makedirs(input_dir)

        output_dir = os.path.join(temp_dir, 'output_dir')
        os.makedirs(output_dir)
        pathlib.Path(os.path.join(input_dir, 'example_point1.tif')).touch()
        pathlib.Path(os.path.join(input_dir, 'example_point2.tif')).touch()
        pathlib.Path(os.path.join(input_dir, 'example_point3.tif')).touch()

        create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                               points=['example_point1', 'example_point2'])

        # make sure DeepCell (.zip) output exists
        assert os.path.exists(os.path.join(output_dir, 'example_output.zip'))

        # DeepCell output .zip file should be extracted
        assert os.path.exists(os.path.join(output_dir, 'example_point1_feature_0.tif'))
        assert os.path.exists(os.path.join(output_dir, 'example_point2_feature_0.tif'))

        # /points.zip file should be removed from input folder after completion
        assert not os.path.isfile(os.path.join(input_dir, 'points.zip'))

        pathlib.Path(os.path.join(input_dir, 'points.zip')).touch()
        # Warning should be displayed if /points.zip file exists (will be overwritten)
        with pytest.warns(UserWarning):
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   points=['example_point1'])

        # DeepCell output .tif file does not exist for some point
        with pytest.warns(UserWarning):
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   suffix='_other_suffix', points=['example_point1'])
        with pytest.warns(UserWarning):
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   points=['example_point1', 'example_point2', 'example_point3'])

        # ValueError should be raised if .tif file does not exists for some point in points
        with pytest.raises(ValueError):
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   points=['example_point1', 'other_point'])