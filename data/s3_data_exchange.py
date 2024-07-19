import boto3
import os
from pathlib import Path
from tqdm import tqdm


def upload_file_s3(
    checkpoint_file, s3_checkpoint_file, bucket_name="jettrain-experiments"
):
    # Initialize the S3 client
    aws_access_key_id, aws_secret_access_key = (
        os.environ["AWS_KEY"],
        os.environ["AWS_SECRET"],
    )
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Iterate over the files in the local directory
    if os.path.isfile(checkpoint_file):
        # Generate the S3 key for the file
        s3_key = s3_checkpoint_file

        # Full path to the local file
        local_file_path = checkpoint_file

        # Upload the file to S3
        s3.upload_file(local_file_path, bucket_name, s3_key)
        print(f"Uploaded {checkpoint_file} to S3://{bucket_name}/{s3_key}")


def download_file_s3(
    local_checkpoint_file, s3_checkpoint_file, bucket_name="jettrain-experiments"
):
    # Initialize the S3 client
    aws_access_key_id, aws_secret_access_key = (
        os.environ["AWS_KEY"],
        os.environ["AWS_SECRET"],
    )
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Ensure the local directory exists
    local_file_directory = os.path.dirname(local_checkpoint_file)
    if not os.path.exists(local_file_directory):
        os.makedirs(local_file_directory)

    # Download the file from S3
    s3.download_file(bucket_name, s3_checkpoint_file, local_checkpoint_file)
    print(
        f"Downloaded {s3_checkpoint_file} from S3://{bucket_name}/{s3_checkpoint_file} to {local_checkpoint_file}"
    )


def upload_directory_s3(
    checkpoint_folder, s3_directory, bucket_name="jettrain-experiments"
):
    # Initialize the S3 client
    aws_access_key_id, aws_secret_access_key = (
        os.environ["AWS_KEY"],
        os.environ["AWS_SECRET"],
    )
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Iterate over the files in the local directory
    for filename in os.listdir(checkpoint_folder):
        # Check if it's a file (and not a directory)
        if os.path.isfile(os.path.join(checkpoint_folder, filename)):
            # Generate the S3 key for the file
            s3_key = s3_directory + filename

            # Full path to the local file
            local_file_path = os.path.join(checkpoint_folder, filename)

            # Upload the file to S3
            s3.upload_file(local_file_path, bucket_name, s3_key)
            print(f"Uploaded {filename} to S3://{bucket_name}/{s3_key}")


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    aws_access_key_id, aws_secret_access_key = (
        os.environ["AWS_KEY"],
        os.environ["AWS_SECRET"],
    )
    s3_resource = boto3.resource(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    s3_folder = s3_folder[len(bucket_name) + 6 :]

    bucket = s3_resource.Bucket(bucket_name)

    local_dir_path = Path(local_dir) if local_dir is not None else Path.cwd()

    for obj in tqdm(bucket.objects.filter(Prefix=s3_folder)):
        obj_key_path = Path(obj.key)
        if obj.key.endswith("/"):
            continue

        target_path = (
            obj_key_path
            if local_dir is None
            else local_dir_path / obj_key_path.relative_to(Path(s3_folder).parent)
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        bucket.download_file(obj.key, str(target_path))

    return str(local_dir_path / Path(s3_folder).name)


def check_if_file_exists(bucket, key):
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    # TODO specify the error
    except:
        return False
