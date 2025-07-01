import pandas as pd
from datetime import datetime, timedelta
import random


def generate_logs():
    """
    Generate synthetic server log data with normal patterns and injected anomalies.
    Creates approximately 200 log entries with timestamp, ip_address, http_status, and request_path.
    """

    common_ips = ["192.168.1.10", "10.0.0.5", "192.168.1.12"]
    ip_weights = [0.6, 0.25, 0.15]

    normal_statuses = [200, 200, 200, 200, 200, 304, 304]

    request_paths = [
        "/index.html",
        "/api/users",
        "/api/data",
        "/login",
        "/dashboard",
        "/static/css/style.css",
        "/static/js/app.js",
        "/favicon.ico",
    ]

    log_data = []

    current_time = datetime(2024, 1, 15, 10, 0, 0)

    total_entries = 200
    anomaly_positions = [25, 50, 75, 120, 150, 175, 190]

    for i in range(total_entries):
        if i in anomaly_positions:
            if i == 25:
                ip = "177.21.55.134"
                status = 200
                path = "/index.html"
                time_increment = random.randint(5, 15)
            elif i == 50:
                ip = random.choices(common_ips, weights=ip_weights)[0]
                status = 500
                path = "/api/data"
                time_increment = random.randint(5, 15)
            elif i == 75:
                ip = random.choices(common_ips, weights=ip_weights)[0]
                status = 404
                path = "/nonexistent-page"
                time_increment = random.randint(5, 15)
            elif i == 120:
                ip = random.choices(common_ips, weights=ip_weights)[0]
                status = random.choice(normal_statuses)
                path = random.choice(request_paths)
                time_increment = 300
            elif i == 150:
                ip = "177.21.55.134"
                status = 500
                path = "/api/users"
                time_increment = random.randint(5, 15)
            elif i == 175:
                ip = random.choices(common_ips, weights=ip_weights)[0]
                status = 404
                path = "/missing-resource"
                time_increment = random.randint(5, 15)
            else:
                ip = "177.21.55.134"
                status = 200
                path = "/suspicious-endpoint"
                time_increment = 180
        else:
            ip = random.choices(common_ips, weights=ip_weights)[0]
            status = random.choice(normal_statuses)
            path = random.choice(request_paths)
            time_increment = random.randint(5, 15)

        log_entry = {
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ip_address": ip,
            "http_status": status,
            "request_path": path,
        }

        log_data.append(log_entry)

        current_time += timedelta(seconds=time_increment)

    df = pd.DataFrame(log_data)
    df.to_csv("server_logs.csv", index=False)

    print(f"Generated {len(log_data)} log entries and saved to 'server_logs.csv'")
    print("\nSample of generated data:")
    print(df.head(10))
    print(f"\nAnomalies injected at positions: {anomaly_positions}")
    print(f"\n Anomalous entries preview:")
    for pos in anomaly_positions[:3]:
        print(f"Position {pos}: {df.iloc[pos].to_dict()}")


if __name__ == "__main__":
    generate_logs()
