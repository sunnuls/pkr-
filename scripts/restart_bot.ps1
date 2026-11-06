# Kill the running play_phone.py; the play_phone.ps1 window restarts it with fresh code.
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*play_phone*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "restarted $($_.ProcessId)" }
