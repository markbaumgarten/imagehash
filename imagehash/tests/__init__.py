from __future__ import (absolute_import, division, print_function)

import os
import os.path
import unittest

from PIL import Image

import imagehash


class TestImageHash(unittest.TestCase):
    @staticmethod
    def get_data_image(fname=None):
        if fname is None:
            fname = 'imagehash.png'
        dname = os.path.abspath(os.path.dirname(__file__))
        target = os.path.join(dname, 'data', fname)
        if not os.path.isfile(target):
            emsg = 'Unknown test image file: {!r}'
            raise ValueError(emsg.format(target))
        return Image.open(target)

    def check_hash_algorithm(self, func, image):
        original_hash = func(image)
        rotate_image = image.rotate(-1)
        rotate_hash = func(rotate_image)
        distance = original_hash - rotate_hash
        emsg = ('slightly rotated image should have '
                'similar hash {} {} {}'.format(original_hash, rotate_hash,
                                               distance))
        self.assertTrue(distance <= 10, emsg)
        rotate_image = image.rotate(-90)
        rotate_hash = func(rotate_image)
        emsg = ('rotated image should have different '
                'hash {} {}'.format(original_hash, rotate_hash))
        self.assertNotEqual(original_hash, rotate_hash, emsg)
        distance = original_hash - rotate_hash
        emsg = ('rotated image should have larger different '
                'hash {} {} {}'.format(original_hash, rotate_hash,
                                       distance))
        self.assertTrue(distance > 10, emsg)

    def check_hash_length(self, func, image, sizes=range(2,21)):
        for hash_size in sizes:
            image_hash = func(image, hash_size=hash_size)
            emsg = 'hash_size={} is not respected'.format(hash_size)
            self.assertEqual(image_hash.hash.size, hash_size**2, emsg)

    def check_all_permutations(self, func, image, sizes=range(2,8)):
        for hash_size in sizes:
            image_hash = func(image, hash_size)
            perms = imagehash.get_all_permutations(str(image_hash))
            #for perm in perms:
            #    print(perm)

            #emsg = 'stringified hash {} != original hash {}'.format(other_hash,
            #                                                        image_hash)
    
    def check_all_permutations_from_int(self, func, image, sizes=range(2,8)):
        for hash_size in sizes:
            image_hash = func(image, hash_size)
            perms = imagehash.get_all_permutations_from_int(image_hash.to_int())
            print(perms)


    def check_hash_stored(self, func, image, sizes=range(2,21)):
        for hash_size in sizes:
            image_hash = func(image, hash_size)
            other_hash = imagehash.hex_to_hash(str(image_hash))
            emsg = 'stringified hash {} != original hash {}'.format(other_hash,
                                                                    image_hash)
            self.assertEqual(image_hash, other_hash, emsg)
            distance = image_hash - other_hash
            emsg = ('unexpected hamming distance {}: original hash {} '
                    '- stringified hash {}'.format(distance, image_hash,
                                                   other_hash))
            self.assertEqual(distance, 0, emsg)

    def check_hash_size(self, func, image, sizes=range(-1,2)):
        for hash_size in sizes:
            with self.assertRaises(ValueError):
                func(image, hash_size)

    def test_get_all_permutations_int_1(self):
        p1 = imagehash.get_all_permutations_from_int(1)
        assert p1 == {72057594037927936, 1, 9223372036854775808, 128}
        p1 = imagehash.get_all_permutations_from_int(0)
        assert p1 == {0}
        p1 = imagehash.get_all_permutations_from_int(1234567)
        assert p1 == {216735757837207297, 4746209, 74031217510449667, 9286633539481125056, 9788030631345651712, 13853072866121146496, 16243155645879025664, 1234567}

    def test_get_all_permutations_hex_1(self):
        p1 = imagehash.get_all_permutations("00c0e0e0c0c0f0f0")
        assert p1 == {'0003070703030f0f', 'f0f0c0c0e0e0c000', '00000000c0ccfefe', '7f7f330300000000', '0000000003337f7f', '0f0f030307070300', 'fefeccc000000000', '00c0e0e0c0c0f0f0'}
