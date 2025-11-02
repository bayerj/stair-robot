setup ip port:
    ssh root@{{ip}} -p {{port}} -i ~/.ssh/id_ed25519 "ls /"

# Internal function to get SSH command with proper auth
_get-ssh-command:
    #!/usr/bin/env bash
    set -euo pipefail

    # Get the SSH connection command output
    output=$(runpodctl ssh connect)

    # Extract pod ID from the comment (format: # pod { id: "xyz", ... })
    pod_id=$(echo "$output" | sed -n 's/.*id: "\([^"]*\)".*/\1/p')

    if [ -z "$pod_id" ]; then
        echo "Error: Could not extract pod ID from runpodctl output" >&2
        exit 1
    fi

    # Extract the SSH connection details and add the SSH key
    # Strip the comment (everything after #) from the SSH command
    ssh_command_base=$(echo "$output" | grep "^ssh " | sed 's/#.*//')
    ssh_command="$ssh_command_base -i ~/.ssh/id_ed25519 -o PasswordAuthentication=no -A"

    # Add BatchMode to prevent password prompts during test
    ssh_command_test="$ssh_command -o BatchMode=yes -o ConnectTimeout=10 -o PreferredAuthentications=publickey"

    # Try to connect with key-only auth (will fail if key not authorized)
    set +e  # Temporarily disable exit on error for the test
    eval "$ssh_command_test exit" >/dev/null 2>&1
    test_exit_code=$?
    set -e  # Re-enable exit on error

    if [ $test_exit_code -ne 0 ]; then
        echo "Error: SSH key not authorized yet. Please follow these steps:" >&2
        echo "1. Go to: https://console.runpod.io/pods?id=$pod_id" >&2
        echo "2. Log in and open the web terminal" >&2
        echo "3. Execute: mkdir -p ~/.ssh && cp /workspace/ssh/authorized_keys ~/.ssh/ && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys" >&2
        exit 1
    fi

    # Output the SSH command for use by other recipes
    echo "$ssh_command"

runpod-connect:
    #!/usr/bin/env bash
    set -euo pipefail
    ssh_command=$(just _get-ssh-command)
    echo "Connecting to pod..."
    eval "exec $ssh_command"

# Execute a command on the runpod
runpod-exec +args:
    #!/usr/bin/env bash
    set -euo pipefail
    ssh_command=$(just _get-ssh-command)
    eval "$ssh_command" "{{ args }}"
