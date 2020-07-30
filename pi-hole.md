# Setup Pi-hole with Docker on Ubuntu

## Modify systemd-resolvd
Modern Ubuntu comes with systemd-resolved which is configured by default to implement a caching DNS stub resolver. This will prevent pi-hole from listtening on port 53.

- Set `DNSStubListener=no` on /etc/systemd/resolved.conf
- `sudo systemctl reload-or-restart systemd-resolved`
- `sudo rm /etc/resolv.conf && ln -s /run/systemd/resolve/resolv.conf /etc/resolv.conf`
- (previously was resolv.conf -> ../run/systemd/resolve/stub-resolv.conf)
- `/run/systemd/resolve/resolv.conf`  is managed by systemd; use 127.0.0.1 as DNS server (set on the router, or dhcp server or manually in netplan/nework manager).

## For DHCP server (optional)
- Enable port in UFW: `sudo ufw allow bootps`
  
## Setup systemd service
```/etc/systemd/system/pihole.service
[Unit]
Description=pihole
Requires=docker.service
After=docker.service

[Service]
User=cxz
WorkingDirectory=/home/cxz/pihole
Restart=always
ExecStart=/usr/bin/docker-compose -f /home/cxz/pihole/docker-compose.yml up
ExecStop=/usr/bin/docker-compose -f /home/cxz/pihole/docker-compose.yml down

[Install]
WantedBy=multi-user.target                          
```

- `systemctl daemon-reload`
- `systemctl enable pihole`
- `systemctl status pihole`
- `journalctl -xe -u pihole`
  
## Logs
- See logs with `docker logs pihole`

