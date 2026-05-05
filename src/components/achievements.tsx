import { Variants, motion } from 'framer-motion';
import SectionWrapper from './sectionWrapper';
import { fadeIn, textVariant } from '../utils/motion';
import { achievements } from '../constants/index';
import Image, { StaticImageData } from 'next/image';

const AchievementCard = ({
  index,
  achievement,
  name,
  designation,
  image,
  link,
}: {
  index: number;
  achievement: string;
  name: string;
  designation: string;
  image: StaticImageData;
  link: string;
}) => (
  <motion.div
    variants={fadeIn('up', 'spring', index * 0.4, 0.75) as Variants}
    className='pointer-events-auto relative w-full rounded-2xl border border-accent/40 bg-background p-8 shadow-card sm:w-[320px]'
  >
    <div className='absolute right-6 top-6 flex justify-end'>
      <a
        href={link}
        target='_blank'
        rel='noreferrer'
        aria-label={`Open ${name}`}
        className='flex h-7 w-7 cursor-pointer items-center justify-center text-text transition-transform duration-200 hover:-translate-y-[2px] hover:text-primary focus:-translate-y-[2px] focus:text-primary'
      >
        <svg height='1em' viewBox='0 0 512 512' className='text-current'>
          <path
            fill='currentColor'
            d='M320 0c-17.7 0-32 14.3-32 32s14.3 32 32 32h82.7L201.4 265.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0L448 109.3V192c0 17.7 14.3 32 32 32s32-14.3 32-32V32c0-17.7-14.3-32-32-32H320zM80 32C35.8 32 0 67.8 0 112V432c0 44.2 35.8 80 80 80H400c44.2 0 80-35.8 80-80V320c0-17.7-14.3-32-32-32s-32 14.3-32 32V432c0 8.8-7.2 16-16 16H80c-8.8 0-16-7.2-16-16V112c0-8.8 7.2-16 16-16H192c17.7 0 32-14.3 32-32s-14.3-32-32-32H80z'
          />
        </svg>
      </a>
    </div>

    <p className='text-[42px] font-black leading-none text-primary'>&#42;</p>

    <p className='mt-3 text-[16px] tracking-wide text-text'>{achievement}</p>

    <div className='mt-7 flex items-center justify-between gap-3'>
      <div className='flex flex-1 flex-col'>
        <p className='text-[15px] font-semibold text-text'>
          <span className='text-accent'>@</span> {name}
        </p>
        <p className='mt-1 text-[12px] text-text opacity-70'>{designation}</p>
      </div>

      <Image src={image} alt={name} className='h-10 w-10 rounded-full object-cover' />
    </div>
  </motion.div>
);

const Achievements = () => {
  return (
    <>
      <motion.div variants={textVariant() as Variants}>
        <p className='section-subtitle'>Recognition</p>
        <h2 className='section-title'>Achievements.</h2>
      </motion.div>

      <div className='mt-10 flex flex-wrap justify-center gap-6 sm:justify-start'>
        {achievements.map((achievement, index) => (
          <AchievementCard key={achievement.name} index={index} {...achievement} />
        ))}
      </div>
    </>
  );
};

export default SectionWrapper(Achievements, 'achievements', { tinted: true });
