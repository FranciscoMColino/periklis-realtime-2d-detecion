from setuptools import find_packages, setup

package_name = 'periklis_yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools',
                      'numpy',
                      'sensor_msgs_py',
                      'rclpy'
                      ],
    zip_safe=True,
    maintainer='colino',
    maintainer_email='francisco.m.colino@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cv2_image_viz= periklis_yolo.cv2_image_viz:main',
            'yolo_mask_3d_viz= periklis_yolo.yolo_mask_3d_viz:main',
            'main_yolo_to_3d_posetf_pub= periklis_yolo.main_yolo_to_3d_posetf_pub:main',
        ],
    },
)
