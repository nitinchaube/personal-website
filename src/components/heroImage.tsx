import heroPhoto from '../../public/photos/hero/hero.png';
import Image from 'next/image';

const HeroImage = () => {
  return (
    <div className='flex justify-center'>
      <div className='relative h-[160px] w-[160px] rounded-full bg-gradient-120 from-primary to-accent p-[3px] shadow-xl shadow-primary/25 transition-transform duration-500 hover:scale-[1.03] xs:h-[190px] xs:w-[190px] sm:h-[220px] sm:w-[220px] lg:h-[260px] lg:w-[260px]'>
        <div className='relative h-full w-full overflow-hidden rounded-full bg-secondary'>
          <Image
            src={heroPhoto}
            alt='Nitin Chaube'
            fill
            sizes='(min-width: 1024px) 260px, (min-width: 640px) 220px, (min-width: 480px) 190px, 160px'
            className='object-cover object-top'
            priority
          />
        </div>
      </div>
    </div>
  );
};

export default HeroImage;
