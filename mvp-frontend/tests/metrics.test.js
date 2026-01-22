import { describe, expect, it } from 'vitest';
import { shuffleArray } from '../src/utils/shuffle';

describe('shuffleArray', () => {
  it('returns a new array with same length', () => {
    const source = [1, 2, 3, 4];
    const result = shuffleArray(source);
    expect(result).toHaveLength(source.length);
    expect(result).not.toBe(source);
  });
});
