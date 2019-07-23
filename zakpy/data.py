import gzip
import pandas as pd
import boto3
from io import BytesIO


def s3_to_df(bucket, key, gzipped=True):
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    raw = BytesIO(obj['Body'].read())
    if gzipped:
        f = gzip.GzipFile(fileobj=raw, mode='r')
        df = pd.read_csv(f)
    else:
        df = pd.read_csv(raw)
    return df


def df_to_s3(bucket, key, df, index=False, header=False):
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=index, header=header)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, key).put(Body=csv_buffer.getvalue())