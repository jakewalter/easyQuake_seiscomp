Example usage

Create a jobs dir and create a job to run detection_continuous:

```bash
mkdir -p /tmp/sceasyquake/jobs
sceasyquake-client job --job-dir /tmp/sceasyquake/jobs --function detection_continuous --params '{"dirname":"20250101","project_folder":"/data/id","project_code":"test","machine":true,"local":true}'

# start worker
sceasyquake-worker --job-dir /tmp/sceasyquake/jobs --python-cmd python
```
