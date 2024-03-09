import os

ram_size = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # in Bytes
ram_size_gb = ram_size / (1024 ** 3)  # Convert to GB

print(f"Total RAM size: {ram_size_gb:.2f} GB")
