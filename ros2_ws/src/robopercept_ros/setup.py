from glob import glob

from setuptools import setup

package_name = "robopercept_ros"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FARHAT Mohamad",
    description="ROS 2 wrapper of RoboPercept",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = robopercept_ros.perception_node:main",
            "speed_governor = robopercept_ros.speed_governor:main",
        ],
    },
)
